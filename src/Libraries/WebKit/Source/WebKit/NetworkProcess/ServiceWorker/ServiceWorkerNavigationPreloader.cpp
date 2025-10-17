/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
#include "config.h"
#include "ServiceWorkerNavigationPreloader.h"

#include "DownloadManager.h"
#include "Logging.h"
#include "NetworkCache.h"
#include "NetworkLoad.h"
#include "NetworkSession.h"
#include "PrivateRelayed.h"
#include <WebCore/HTTPStatusCodes.h>
#include <WebCore/NavigationPreloadState.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(ServiceWorkerNavigationPreloader);

ServiceWorkerNavigationPreloader::ServiceWorkerNavigationPreloader(NetworkSession& session, NetworkLoadParameters&& parameters, const WebCore::NavigationPreloadState& state, bool shouldCaptureExtraNetworkLoadMetric)
    : m_session(session)
    , m_parameters(WTFMove(parameters))
    , m_state(state)
    , m_shouldCaptureExtraNetworkLoadMetrics(shouldCaptureExtraNetworkLoadMetrics())
    , m_startTime(MonotonicTime::now())
{
    RELEASE_LOG(ServiceWorker, "ServiceWorkerNavigationPreloader::ServiceWorkerNavigationPreloader %p", this);
    start();
}

void ServiceWorkerNavigationPreloader::start()
{
    if (m_isStarted)
        return;
    m_isStarted = true;

    if (!m_session) {
        didFailLoading(ResourceError { errorDomainWebKitInternal, 0, { }, "No session for preload"_s });
        return;
    }

    if (m_session->cache()) {
        NetworkCache::GlobalFrameID globalID { *m_parameters.webPageProxyID, *m_parameters.webPageID, *m_parameters.webFrameID };
        m_session->cache()->retrieve(m_parameters.request, globalID, m_parameters.isNavigatingToAppBoundDomain, m_parameters.allowPrivacyProxy, m_parameters.advancedPrivacyProtections, [this, weakThis = WeakPtr { *this }](auto&& entry, auto&&) mutable {
            CheckedPtr checkedThis = weakThis.get();
            if (!checkedThis || m_isCancelled)
                return;

            if (entry && !entry->needsValidation()) {
                loadWithCacheEntry(*entry);
                return;
            }

            m_parameters.request.setCachePolicy(ResourceRequestCachePolicy::RefreshAnyCacheData);
            if (entry) {
                m_cacheEntry = WTFMove(entry);

                auto eTag = m_cacheEntry->response().httpHeaderField(HTTPHeaderName::ETag);
                if (!eTag.isEmpty())
                    m_parameters.request.setHTTPHeaderField(HTTPHeaderName::IfNoneMatch, eTag);

                auto lastModified = m_cacheEntry->response().httpHeaderField(HTTPHeaderName::LastModified);
                if (!lastModified.isEmpty())
                    m_parameters.request.setHTTPHeaderField(HTTPHeaderName::IfModifiedSince, lastModified);
            }

            if (!m_session) {
                didFailLoading(ResourceError { ResourceError::Type::Cancellation });
                return;
            }
            loadFromNetwork();
        });
        return;
    }
    loadFromNetwork();
}

ServiceWorkerNavigationPreloader::~ServiceWorkerNavigationPreloader()
{
}

void ServiceWorkerNavigationPreloader::cancel()
{
    m_isCancelled = true;
    if (m_responseCompletionHandler)
        m_responseCompletionHandler(PolicyAction::Ignore);
    if (m_networkLoad)
        m_networkLoad->cancel();
}

void ServiceWorkerNavigationPreloader::loadWithCacheEntry(NetworkCache::Entry& entry)
{
    didReceiveResponse(ResourceResponse { entry.response() }, PrivateRelayed::No, [body = RefPtr { entry.buffer() }, weakThis = WeakPtr { *this }](auto) mutable {
        if (!weakThis || weakThis->m_isCancelled)
            return;

        size_t size  = 0;
        if (body) {
            size = body->size();
            weakThis->didReceiveBuffer(body.releaseNonNull(), size);
            if (!weakThis)
                return;
        }

        NetworkLoadMetrics networkLoadMetrics;
        networkLoadMetrics.markComplete();
        networkLoadMetrics.responseBodyBytesReceived = 0;
        networkLoadMetrics.responseBodyDecodedSize = size;
        if (weakThis->shouldCaptureExtraNetworkLoadMetrics()) {
            auto additionalMetrics = WebCore::AdditionalNetworkLoadMetricsForWebInspector::create();
            additionalMetrics->requestHeaderBytesSent = 0;
            additionalMetrics->requestBodyBytesSent = 0;
            additionalMetrics->responseHeaderBytesReceived = 0;
            networkLoadMetrics.additionalNetworkLoadMetricsForWebInspector = WTFMove(additionalMetrics);
        }
        weakThis->didFinishLoading(networkLoadMetrics);
    });
    didComplete();
}

void ServiceWorkerNavigationPreloader::loadFromNetwork()
{
    ASSERT(m_session);
    RELEASE_LOG(ServiceWorker, "ServiceWorkerNavigationPreloader::loadFromNetwork %p", this);

    if (m_state.enabled)
        m_parameters.request.addHTTPHeaderField(HTTPHeaderName::ServiceWorkerNavigationPreload, m_state.headerValue);

    m_networkLoad = NetworkLoad::create(*this, WTFMove(m_parameters), *m_session);
    m_networkLoad->start();
}

void ServiceWorkerNavigationPreloader::willSendRedirectedRequest(ResourceRequest&&, ResourceRequest&&, ResourceResponse&& response, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler)
{
    didReceiveResponse(WTFMove(response), PrivateRelayed::No, [weakThis = WeakPtr { *this }, completionHandler = WTFMove(completionHandler)](auto) mutable {
        completionHandler({ });
        if (weakThis)
            weakThis->didComplete();
    });
}

void ServiceWorkerNavigationPreloader::didReceiveResponse(ResourceResponse&& response, PrivateRelayed, ResponseCompletionHandler&& completionHandler)
{
    RELEASE_LOG(ServiceWorker, "ServiceWorkerNavigationPreloader::didReceiveResponse %p", this);

    m_didReceiveResponseOrError = true;

    if (response.isRedirection())
        response.setTainting(ResourceResponse::Tainting::Opaqueredirect);

    if (response.httpStatusCode() == httpStatus304NotModified && m_cacheEntry) {
        auto cacheEntry = WTFMove(m_cacheEntry);
        loadWithCacheEntry(*cacheEntry);
        completionHandler(PolicyAction::Ignore);
        return;
    }

    m_response = WTFMove(response);
    m_response.setRedirected(false);
    m_responseCompletionHandler = WTFMove(completionHandler);

    if (auto callback = std::exchange(m_responseCallback, { }))
        callback();
}

void ServiceWorkerNavigationPreloader::didReceiveBuffer(const FragmentedSharedBuffer& buffer, uint64_t reportedEncodedDataLength)
{
    if (m_bodyCallback)
        m_bodyCallback(RefPtr { &buffer }, reportedEncodedDataLength);
}

void ServiceWorkerNavigationPreloader::didFinishLoading(const NetworkLoadMetrics& networkLoadMetrics)
{
    RELEASE_LOG(ServiceWorker, "ServiceWorkerNavigationPreloader::didFinishLoading %p", this);

    m_networkLoadMetrics = networkLoadMetrics;
    didComplete();
}

void ServiceWorkerNavigationPreloader::didFailLoading(const ResourceError& error)
{
    RELEASE_LOG(ServiceWorker, "ServiceWorkerNavigationPreloader::didFailLoading %p", this);

    m_didReceiveResponseOrError = true;
    m_error = error;
    didComplete();
}

void ServiceWorkerNavigationPreloader::didComplete()
{
    m_networkLoad = nullptr;

    auto responseCallback = std::exchange(m_responseCallback, { });
    auto bodyCallback = std::exchange(m_bodyCallback, { });

    // After calling responseCallback or bodyCallback, |this| might be destroyed.
    if (responseCallback)
        responseCallback();

    if (bodyCallback)
        bodyCallback({ }, 0);
}

void ServiceWorkerNavigationPreloader::waitForResponse(ResponseCallback&& callback)
{
    if (!m_error.isNull()) {
        callback();
        return;
    }

    if (m_responseCompletionHandler) {
        callback();
        return;
    }

    m_responseCallback = WTFMove(callback);
}

void ServiceWorkerNavigationPreloader::waitForBody(BodyCallback&& callback)
{
    if (!m_error.isNull() || !m_responseCompletionHandler) {
        callback({ }, 0);
        return;
    }

    ASSERT(!m_response.isNull());
    m_bodyCallback = WTFMove(callback);
    m_responseCompletionHandler(PolicyAction::Use);
}

bool ServiceWorkerNavigationPreloader::convertToDownload(DownloadManager& manager, DownloadID downloadID, const WebCore::ResourceRequest& request, const WebCore::ResourceResponse& response)
{
    if (!m_networkLoad)
        return false;

    auto networkLoad = std::exchange(m_networkLoad, nullptr);
    manager.convertNetworkLoadToDownload(downloadID, networkLoad.releaseNonNull(), WTFMove(m_responseCompletionHandler), { }, request, response);
    return true;
}

} // namespace WebKit
