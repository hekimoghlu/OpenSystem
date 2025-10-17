/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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

#include "DownloadID.h"
#include "NetworkCacheEntry.h"
#include "NetworkLoadClient.h"
#include "NetworkLoadParameters.h"
#include <WebCore/NavigationPreloadState.h>
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class NetworkLoadMetrics;
class FragmentedSharedBuffer;
}

namespace WebKit {

class DownloadManager;
class NetworkLoad;
class NetworkSession;

class ServiceWorkerNavigationPreloader final : public NetworkLoadClient, public CanMakeWeakPtr<ServiceWorkerNavigationPreloader>, public CanMakeCheckedPtr<ServiceWorkerNavigationPreloader> {
    WTF_MAKE_TZONE_ALLOCATED(ServiceWorkerNavigationPreloader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ServiceWorkerNavigationPreloader);
public:
    ServiceWorkerNavigationPreloader(NetworkSession&, NetworkLoadParameters&&, const WebCore::NavigationPreloadState&, bool shouldCaptureExtraNetworkLoadMetrics);
    ~ServiceWorkerNavigationPreloader();

    void cancel();

    using ResponseCallback = Function<void()>;
    void waitForResponse(ResponseCallback&&);
    using BodyCallback = Function<void(RefPtr<const WebCore::FragmentedSharedBuffer>&&, uint64_t reportedEncodedDataLength)>;
    void waitForBody(BodyCallback&&);

    const WebCore::ResourceError& error() const { return m_error; }
    const WebCore::ResourceResponse& response() const { return m_response; }
    const WebCore::NetworkLoadMetrics& networkLoadMetrics() const { return m_networkLoadMetrics; }
    bool isServiceWorkerNavigationPreloadEnabled() const { return m_state.enabled; }
    bool didReceiveResponseOrError() const { return m_didReceiveResponseOrError; }

    bool convertToDownload(DownloadManager&, DownloadID, const WebCore::ResourceRequest&, const WebCore::ResourceResponse&);

    MonotonicTime startTime() const { return m_startTime; }

private:
    // NetworkLoadClient.
    void didSendData(uint64_t bytesSent, uint64_t totalBytesToBeSent) final { }
    bool isSynchronous() const final { return false; }
    bool isAllowedToAskUserForCredentials() const final { return false; }
    void willSendRedirectedRequest(WebCore::ResourceRequest&&, WebCore::ResourceRequest&& redirectRequest, WebCore::ResourceResponse&& redirectResponse, CompletionHandler<void(WebCore::ResourceRequest&&)>&&) final;
    void didReceiveResponse(WebCore::ResourceResponse&&, PrivateRelayed, ResponseCompletionHandler&&) final;
    void didReceiveBuffer(const WebCore::FragmentedSharedBuffer&, uint64_t reportedEncodedDataLength) final;
    void didFinishLoading(const WebCore::NetworkLoadMetrics&) final;
    void didFailLoading(const WebCore::ResourceError&) final;
    bool shouldCaptureExtraNetworkLoadMetrics() const final { return m_shouldCaptureExtraNetworkLoadMetrics; }

    void start();
    void loadWithCacheEntry(NetworkCache::Entry&);
    void loadFromNetwork();
    void didComplete();

    RefPtr<NetworkLoad> m_networkLoad;
    WeakPtr<NetworkSession> m_session;

    NetworkLoadParameters m_parameters;
    WebCore::NavigationPreloadState m_state;

    std::unique_ptr<NetworkCache::Entry> m_cacheEntry;
    WebCore::NetworkLoadMetrics m_networkLoadMetrics;
    WebCore::ResourceResponse m_response;
    ResponseCompletionHandler m_responseCompletionHandler;
    WebCore::ResourceError m_error;

    ResponseCallback m_responseCallback;
    BodyCallback m_bodyCallback;

    bool m_isStarted { false };
    bool m_isCancelled { false };
    bool m_shouldCaptureExtraNetworkLoadMetrics { false };
    bool m_didReceiveResponseOrError { false };
    MonotonicTime m_startTime;
};

} // namespace WebKit
