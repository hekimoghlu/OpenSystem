/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

#include "NetworkDataTask.h"
#include "NetworkLoadParameters.h"
#include <WebCore/CurlRequestClient.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/ProtectionSpace.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/ShouldRelaxThirdPartyCookieBlocking.h>
#include <wtf/FileSystem.h>
#include <wtf/MonotonicTime.h>

namespace WebCore {
class CurlRequest;
class SharedBuffer;
}

namespace WebKit {

class NetworkDataTaskCurl final : public NetworkDataTask, public WebCore::CurlRequestClient {
public:
    static Ref<NetworkDataTask> create(NetworkSession& session, NetworkDataTaskClient& client, const NetworkLoadParameters& parameters)
    {
        return adoptRef(*new NetworkDataTaskCurl(session, client, parameters));
    }

    ~NetworkDataTaskCurl();

    void ref() const final { NetworkDataTask::ref(); }
    void deref() const final { NetworkDataTask::deref(); }

private:
    enum class RequestStatus {
        NewRequest,
        ReusedRequest
    };

    NetworkDataTaskCurl(NetworkSession&, NetworkDataTaskClient&, const NetworkLoadParameters&);

    void cancel() override;
    void resume() override;
    void invalidateAndCancel() override;
    NetworkDataTask::State state() const override;
    void setTimingAllowFailedFlag() final;

    Ref<WebCore::CurlRequest> createCurlRequest(WebCore::ResourceRequest&&, RequestStatus = RequestStatus::NewRequest);
    void curlDidSendData(WebCore::CurlRequest&, unsigned long long, unsigned long long) override;
    void curlDidReceiveResponse(WebCore::CurlRequest&, WebCore::CurlResponse&&) override;
    void curlDidReceiveData(WebCore::CurlRequest&, Ref<WebCore::SharedBuffer>&&) override;
    void curlDidComplete(WebCore::CurlRequest&, WebCore::NetworkLoadMetrics&&) override;
    void curlDidFailWithError(WebCore::CurlRequest&, WebCore::ResourceError&&, WebCore::CertificateInfo&&) override;

    void invokeDidReceiveResponse();

    bool shouldStartHTTPRedirection();
    bool shouldRedirectAsGET(const WebCore::ResourceRequest&, bool crossOrigin);
    void willPerformHTTPRedirection();

    void tryHttpAuthentication(WebCore::AuthenticationChallenge&&);
    void tryProxyAuthentication(WebCore::AuthenticationChallenge&&);
    void tryServerTrustEvaluation(WebCore::AuthenticationChallenge&&);
    void restartWithCredential(const WebCore::ProtectionSpace&, const WebCore::Credential&);

    void appendCookieHeader(WebCore::ResourceRequest&);
    void handleCookieHeaders(const WebCore::ResourceRequest&, const WebCore::CurlResponse&);

    bool isThirdPartyRequest(const WebCore::ResourceRequest&);
    bool shouldBlockCookies(const WebCore::ResourceRequest&);
    void blockCookies();
    void unblockCookies();

    void updateNetworkLoadMetrics(WebCore::NetworkLoadMetrics&);

    void setPendingDownloadLocation(const String&, SandboxExtension::Handle&&, bool /*allowOverwrite*/) override;
    String suggestedFilename() const override;
    void deleteDownloadFile();

    Markable<WebCore::FrameIdentifier> m_frameID;
    Markable<WebCore::PageIdentifier> m_pageID;
    Markable<WebPageProxyIdentifier> m_webPageProxyID;
    RefPtr<WebCore::SecurityOrigin> m_sourceOrigin;

    State m_state { State::Suspended };

    RefPtr<WebCore::CurlRequest> m_curlRequest;
    WebCore::ResourceResponse m_response;
    unsigned m_redirectCount { 0 };
    unsigned m_authFailureCount { 0 };

    bool m_allowOverwriteDownload { false };
    FileSystem::PlatformFileHandle m_downloadDestinationFile { FileSystem::invalidPlatformFileHandle };

    bool m_blockingCookies { false };

    MonotonicTime m_startTime;
    bool m_failsTAOCheck { false };
    bool m_hasCrossOriginRedirect { false };
};

} // namespace WebKit
