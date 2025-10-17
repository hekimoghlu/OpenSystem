/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 27, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#include "CurlFormDataStream.h"
#include "CurlMultipartHandle.h"
#include "CurlMultipartHandleClient.h"
#include "CurlRequestSchedulerClient.h"
#include "CurlResponse.h"
#include "ProtectionSpace.h"
#include "ResourceRequest.h"
#include <wtf/MonotonicTime.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CurlRequestClient;
class NetworkLoadMetrics;
class ResourceError;
class FragmentedSharedBuffer;

class CurlRequest final : public ThreadSafeRefCounted<CurlRequest>, public CurlRequestSchedulerClient, public CurlMultipartHandleClient, public CanMakeThreadSafeCheckedPtr<CurlRequest> {
    WTF_MAKE_TZONE_ALLOCATED(CurlRequest);
    WTF_MAKE_NONCOPYABLE(CurlRequest);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(CurlRequest);
public:
    enum class CaptureNetworkLoadMetrics : uint8_t {
        Basic,
        Extended
    };

    static Ref<CurlRequest> create(const ResourceRequest& request, CurlRequestClient& client, CaptureNetworkLoadMetrics captureMetrics = CaptureNetworkLoadMetrics::Basic)
    {
        return adoptRef(*new CurlRequest(request, &client, captureMetrics));
    }

    virtual ~CurlRequest();

    WEBCORE_EXPORT void invalidateClient();
    WEBCORE_EXPORT void setAuthenticationScheme(ProtectionSpace::AuthenticationScheme);
    WEBCORE_EXPORT void setUserPass(const String&, const String&);
    bool isServerTrustEvaluationDisabled() { return m_shouldDisableServerTrustEvaluation; }
    void disableServerTrustEvaluation() { m_shouldDisableServerTrustEvaluation = true; }
    void enableLocalhostAlias() { m_localhostAlias = CurlHandle::LocalhostAlias::Enable; }

    WEBCORE_EXPORT void resume();
    WEBCORE_EXPORT void cancel();

    const ResourceRequest& resourceRequest() const { return m_request; }
    bool isCancelled();
    bool isCompletedOrCancelled();
    Seconds timeoutInterval() const;

    const String& user() const { return m_user; }
    const String& password() const { return m_password; }

    // Processing for DidReceiveResponse
    WEBCORE_EXPORT void completeDidReceiveResponse();

private:
    WEBCORE_EXPORT CurlRequest(const ResourceRequest&, CurlRequestClient*, CaptureNetworkLoadMetrics);

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeThreadSafeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeThreadSafeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeThreadSafeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeThreadSafeCheckedPtr::decrementCheckedPtrCount(); }

    void retain() override { ref(); }
    void release() override { deref(); }
    CURL* handle() override { return m_curlHandle ? m_curlHandle->handle() : nullptr; }

    void startWithJobManager();

    void callClient(Function<void(CurlRequest&, CurlRequestClient&)>&&);
    void runOnMainThread(Function<void()>&&);
    void runOnWorkerThreadIfRequired(Function<void()>&&);

    // Transfer processing of Request body, Response header/body
    // Called by worker thread in case of async, main thread in case of sync.
    CURL* setupTransfer() override;
    size_t willSendData(char*, size_t, size_t);
    size_t didReceiveHeader(String&&);
    size_t didReceiveData(std::span<const uint8_t>);
    void didReceiveHeaderFromMultipart(Vector<String>&&) override;
    void didReceiveDataFromMultipart(std::span<const uint8_t>) override;
    void didCompleteFromMultipart() override;
    void didCompleteTransfer(CURLcode) override;
    void didCancelTransfer() override;
    void finalizeTransfer();

    int didReceiveDebugInfo(curl_infotype, std::span<const char>);

    // For setup 
    void appendAcceptLanguageHeader(HTTPHeaderMap&);
    void setupPOST();
    void setupPUT();

    // Processing for DidReceiveResponse
    bool needToInvokeDidReceiveResponse() const { return m_didReceiveResponse && !m_didNotifyResponse; }
    bool needToInvokeDidCancelTransfer() const { return m_didNotifyResponse && !m_didReturnFromNotify && m_mustInvokeCancelTransfer; }
    void invokeDidReceiveResponseForFile(const URL&);
    void invokeDidReceiveResponse(const CurlResponse&, Function<void()>&& completionHandler = { });

    NetworkLoadMetrics networkLoadMetrics();
    std::optional<long long> getContentLength();

    // Callback functions for curl
    static size_t willSendDataCallback(char*, size_t, size_t, void*);
    static size_t didReceiveHeaderCallback(char*, size_t, size_t, void*);
    static size_t didReceiveDataCallback(char*, size_t, size_t, void*);
    static int didReceiveDebugInfoCallback(CURL*, curl_infotype, char*, size_t, void*);


    CurlRequestClient* m_client { };
    Lock m_statusMutex;
    bool m_cancelled { false };
    bool m_completed { false };

    ResourceRequest m_request;
    String m_user;
    String m_password;
    unsigned long m_authType { CURLAUTH_ANY };
    bool m_shouldDisableServerTrustEvaluation { false };
    CurlHandle::LocalhostAlias m_localhostAlias { CurlHandle::LocalhostAlias::Disable };

    std::unique_ptr<CurlHandle> m_curlHandle;
    CurlFormDataStream m_formDataStream;
    std::unique_ptr<CurlMultipartHandle> m_multipartHandle;

    CurlResponse m_response;
    bool m_didStartTransfer { false };
    bool m_didReceiveResponse { false };
    bool m_didNotifyResponse { false };
    bool m_didReturnFromNotify { false };
    bool m_mustInvokeCancelTransfer { false };
    Function<void()> m_responseCompletionHandler;

    bool m_captureExtraMetrics;
    size_t m_requestHeaderSize { 0 };
    HTTPHeaderMap m_requestHeaders;
    MonotonicTime m_performStartTime;
    size_t m_totalReceivedSize { 0 };
};

} // namespace WebCore
