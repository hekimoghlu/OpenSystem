/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
#include "WebURLSchemeTaskProxy.h"

#include "Logging.h"
#include "MessageSenderInlines.h"
#include "URLSchemeTaskParameters.h"
#include "WebFrame.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include "WebURLSchemeHandlerProxy.h"
#include <WebCore/NetworkLoadMetrics.h>
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceLoader.h>
#include <wtf/CompletionHandler.h>

#define WEBURLSCHEMETASKPROXY_RELEASE_LOG_STANDARD_TEMPLATE "[schemeHandler=%" PRIu64 ", webPageID=%" PRIu64 ", frameID=%" PRIu64 ", taskID=%" PRIu64 "] WebURLSchemeTaskProxy::"
#define WEBURLSCHEMETASKPROXY_RELEASE_LOG_STANDARD_PARAMETERS m_urlSchemeHandler->identifier().toUInt64(), pageIDFromWebFrame(m_frame), frameIDFromWebFrame(m_frame), m_identifier.toUInt64()
#define WEBURLSCHEMETASKPROXY_RELEASE_LOG(fmt, ...) RELEASE_LOG(Network, WEBURLSCHEMETASKPROXY_RELEASE_LOG_STANDARD_TEMPLATE fmt, WEBURLSCHEMETASKPROXY_RELEASE_LOG_STANDARD_PARAMETERS, ##__VA_ARGS__)

namespace WebKit {
using namespace WebCore;

#if !RELEASE_LOG_DISABLED
static uint64_t pageIDFromWebFrame(const RefPtr<WebFrame>& frame)
{
    if (frame) {
        if (auto* page = frame->page())
            return page->identifier().toUInt64();
    }
    return 0;
}

static uint64_t frameIDFromWebFrame(const RefPtr<WebFrame>& frame)
{
    if (frame)
        return frame->frameID().object().toUInt64();
    return 0;
}
#endif

WebURLSchemeTaskProxy::WebURLSchemeTaskProxy(WebURLSchemeHandlerProxy& handler, ResourceLoader& loader, WebFrame& frame)
    : m_urlSchemeHandler(handler)
    , m_coreLoader(&loader)
    , m_frame(&frame)
    , m_request(loader.request())
    , m_identifier(*loader.identifier())
{
}

void WebURLSchemeTaskProxy::startLoading()
{
    ASSERT(m_coreLoader);
    ASSERT(m_frame);
    WEBURLSCHEMETASKPROXY_RELEASE_LOG("startLoading");
    Ref urlSchemeHandler = m_urlSchemeHandler.get();
    urlSchemeHandler->page().send(Messages::WebPageProxy::StartURLSchemeTask(URLSchemeTaskParameters { urlSchemeHandler->identifier(), *m_coreLoader->identifier(), m_request, m_frame->info() }));
}

void WebURLSchemeTaskProxy::stopLoading()
{
    ASSERT(m_coreLoader);
    WEBURLSCHEMETASKPROXY_RELEASE_LOG("stopLoading");
    Ref urlSchemeHandler = m_urlSchemeHandler.get();
    urlSchemeHandler->page().send(Messages::WebPageProxy::StopURLSchemeTask(urlSchemeHandler->identifier(), *m_coreLoader->identifier()));
    m_coreLoader = nullptr;
    m_frame = nullptr;

    // This line will result in this being deleted.
    urlSchemeHandler->taskDidStopLoading(*this);
}
    
void WebURLSchemeTaskProxy::didPerformRedirection(WebCore::ResourceResponse&& redirectResponse, WebCore::ResourceRequest&& request, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler)
{
    if (!hasLoader()) {
        completionHandler({ });
        return;
    }

    if (m_waitingForCompletionHandler) {
        WEBURLSCHEMETASKPROXY_RELEASE_LOG("didPerformRedirection: Received redirect during previous redirect processing, queuing it.");
        queueTask([this, protectedThis = Ref { *this }, redirectResponse = WTFMove(redirectResponse), request = WTFMove(request), completionHandler = WTFMove(completionHandler)]() mutable {
            didPerformRedirection(WTFMove(redirectResponse), WTFMove(request), WTFMove(completionHandler));
        });
        return;
    }
    m_waitingForCompletionHandler = true;

    auto innerCompletionHandler = [this, protectedThis = Ref { *this }, originalRequest = request, completionHandler = WTFMove(completionHandler)] (ResourceRequest&& request) mutable {
        m_waitingForCompletionHandler = false;

        completionHandler(WTFMove(request));

        processNextPendingTask();
    };

    m_coreLoader->willSendRequest(WTFMove(request), redirectResponse, WTFMove(innerCompletionHandler));
}

void WebURLSchemeTaskProxy::didReceiveResponse(const ResourceResponse& response)
{
    if (m_waitingForCompletionHandler) {
        WEBURLSCHEMETASKPROXY_RELEASE_LOG("didReceiveResponse: Received response during redirect processing, queuing it.");
        queueTask([this, protectedThis = Ref { *this }, response] {
            didReceiveResponse(response);
        });
        return;
    }
    
    if (!hasLoader())
        return;

    m_waitingForCompletionHandler = true;
    m_coreLoader->didReceiveResponse(response, [this, protectedThis = Ref { *this }] {
        m_waitingForCompletionHandler = false;
        processNextPendingTask();
    });
}

void WebURLSchemeTaskProxy::didReceiveData(const WebCore::SharedBuffer& data)
{
    if (!hasLoader())
        return;

    if (m_waitingForCompletionHandler) {
        WEBURLSCHEMETASKPROXY_RELEASE_LOG("didReceiveData: Received data during response processing, queuing it.");
        queueTask([this, protectedThis = Ref { *this }, data = Ref { data }] {
            didReceiveData(data);
        });
        return;
    }

    Ref protectedThis { *this };
    m_coreLoader->didReceiveData(data, 0, DataPayloadType::DataPayloadBytes);
    processNextPendingTask();
}

void WebURLSchemeTaskProxy::didComplete(const ResourceError& error)
{
    WEBURLSCHEMETASKPROXY_RELEASE_LOG("didComplete");
    if (!hasLoader())
        return;

    if (m_waitingForCompletionHandler) {
        queueTask([this, protectedThis = Ref { *this }, error] {
            didComplete(error);
        });
        return;
    }

    if (error.isNull())
        m_coreLoader->didFinishLoading(NetworkLoadMetrics());
    else
        m_coreLoader->didFail(error);

    m_coreLoader = nullptr;
    m_frame = nullptr;
}

bool WebURLSchemeTaskProxy::hasLoader()
{
    if (m_coreLoader && m_coreLoader->reachedTerminalState()) {
        m_coreLoader = nullptr;
        m_frame = nullptr;
    }

    return m_coreLoader;
}

void WebURLSchemeTaskProxy::processNextPendingTask()
{
    if (!m_queuedTasks.isEmpty())
        m_queuedTasks.takeFirst()();
}

} // namespace WebKit

#undef WEBURLSCHEMETASKPROXY_RELEASE_LOG_STANDARD_TEMPLATE
#undef WEBURLSCHEMETASKPROXY_RELEASE_LOG_STANDARD_PARAMETERS
#undef WEBURLSCHEMETASKPROXY_RELEASE_LOG
