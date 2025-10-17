/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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
#include "WebURLSchemeTask.h"

#include "APIFrameInfo.h"
#include "MessageSenderInlines.h"
#include "URLSchemeTaskParameters.h"
#include "WebErrors.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"
#include "WebURLSchemeHandler.h"

namespace WebKit {
using namespace WebCore;

Ref<WebURLSchemeTask> WebURLSchemeTask::create(WebURLSchemeHandler& handler, WebPageProxy& page, WebProcessProxy& process, PageIdentifier webPageID, URLSchemeTaskParameters&& parameters, SyncLoadCompletionHandler&& syncCompletionHandler)
{
    return adoptRef(*new WebURLSchemeTask(handler, page, process, webPageID, WTFMove(parameters), WTFMove(syncCompletionHandler)));
}

WebURLSchemeTask::WebURLSchemeTask(WebURLSchemeHandler& handler, WebPageProxy& page, WebProcessProxy& process, PageIdentifier webPageID, URLSchemeTaskParameters&& parameters, SyncLoadCompletionHandler&& syncCompletionHandler)
    : m_urlSchemeHandler(handler)
    , m_process(&process)
    , m_resourceLoaderID(parameters.taskIdentifier)
    , m_pageProxyID(page.identifier())
    , m_webPageID(webPageID)
    , m_request(WTFMove(parameters.request))
    , m_frameInfo(API::FrameInfo::create(WTFMove(parameters.frameInfo), &page))
    , m_syncCompletionHandler(WTFMove(syncCompletionHandler))
{
    ASSERT(RunLoop::isMain());
}

WebURLSchemeTask::~WebURLSchemeTask()
{
    ASSERT(RunLoop::isMain());
}

ResourceRequest WebURLSchemeTask::request() const
{
    ASSERT(RunLoop::isMain());
    Locker locker { m_requestLock };
    return m_request;
}

auto WebURLSchemeTask::willPerformRedirection(ResourceResponse&& response, ResourceRequest&& request, Function<void(ResourceRequest&&)>&& completionHandler) -> ExceptionType
{
    ASSERT(RunLoop::isMain());

    if (m_stopped)
        return m_shouldSuppressTaskStoppedExceptions ? ExceptionType::None : ExceptionType::TaskAlreadyStopped;

    if (m_completed)
        return ExceptionType::CompleteAlreadyCalled;

    if (m_dataSent)
        return ExceptionType::DataAlreadySent;

    if (m_responseSent)
        return ExceptionType::RedirectAfterResponse;

    if (m_waitingForRedirectCompletionHandlerCallback)
        return ExceptionType::WaitingForRedirectCompletionHandler;

    if (isSync())
        m_syncResponse = response;

    {
        Locker locker { m_requestLock };
        m_request = request;
    }

    RefPtr page = m_pageProxyID ? WebProcessProxy::webPage(*m_pageProxyID) : nullptr;
    if (!page || !m_process)
        return ExceptionType::None;

    m_waitingForRedirectCompletionHandlerCallback = true;

    Function<void(ResourceRequest&&)> innerCompletionHandler = [protectedThis = Ref { *this }, this, completionHandler = WTFMove(completionHandler)] (ResourceRequest&& request) mutable {
        m_waitingForRedirectCompletionHandlerCallback = false;
        if (!m_stopped && !m_completed)
            completionHandler(WTFMove(request));
    };

    m_process->sendWithAsyncReply(Messages::WebPage::URLSchemeTaskWillPerformRedirection(m_urlSchemeHandler->identifier(), m_resourceLoaderID, response, request), WTFMove(innerCompletionHandler), page->webPageIDInMainFrameProcess());

    return ExceptionType::None;
}

auto WebURLSchemeTask::didPerformRedirection(WebCore::ResourceResponse&& response, WebCore::ResourceRequest&& request) -> ExceptionType
{
    ASSERT(RunLoop::isMain());

    if (m_stopped)
        return m_shouldSuppressTaskStoppedExceptions ? ExceptionType::None : ExceptionType::TaskAlreadyStopped;

    if (m_completed)
        return ExceptionType::CompleteAlreadyCalled;

    if (m_dataSent)
        return ExceptionType::DataAlreadySent;

    if (m_responseSent)
        return ExceptionType::RedirectAfterResponse;

    if (m_waitingForRedirectCompletionHandlerCallback)
        return ExceptionType::WaitingForRedirectCompletionHandler;

    if (isSync())
        m_syncResponse = response;

    {
        Locker locker { m_requestLock };
        m_request = request;
    }

    m_process->send(Messages::WebPage::URLSchemeTaskDidPerformRedirection(m_urlSchemeHandler->identifier(), m_resourceLoaderID, response, request), *m_webPageID);

    return ExceptionType::None;
}

auto WebURLSchemeTask::didReceiveResponse(const ResourceResponse& response) -> ExceptionType
{
    ASSERT(RunLoop::isMain());

    if (m_stopped)
        return m_shouldSuppressTaskStoppedExceptions ? ExceptionType::None : ExceptionType::TaskAlreadyStopped;

    if (m_completed)
        return ExceptionType::CompleteAlreadyCalled;

    if (m_dataSent)
        return ExceptionType::DataAlreadySent;

    if (m_waitingForRedirectCompletionHandlerCallback)
        return ExceptionType::WaitingForRedirectCompletionHandler;

    m_responseSent = true;

    if (isSync())
        m_syncResponse = response;

    m_process->send(Messages::WebPage::URLSchemeTaskDidReceiveResponse(m_urlSchemeHandler->identifier(), m_resourceLoaderID, response), *m_webPageID);
    return ExceptionType::None;
}

auto WebURLSchemeTask::didReceiveData(Ref<SharedBuffer>&& buffer) -> ExceptionType
{
    ASSERT(RunLoop::isMain());

    if (m_stopped)
        return m_shouldSuppressTaskStoppedExceptions ? ExceptionType::None : ExceptionType::TaskAlreadyStopped;

    if (m_completed)
        return ExceptionType::CompleteAlreadyCalled;

    if (!m_responseSent)
        return ExceptionType::NoResponseSent;

    if (m_waitingForRedirectCompletionHandlerCallback)
        return ExceptionType::WaitingForRedirectCompletionHandler;

    m_dataSent = true;

    if (isSync()) {
        m_syncData.append(buffer);
        return ExceptionType::None;
    }

    m_process->send(Messages::WebPage::URLSchemeTaskDidReceiveData(m_urlSchemeHandler->identifier(), m_resourceLoaderID, WTFMove(buffer)), *m_webPageID);
    return ExceptionType::None;
}

auto WebURLSchemeTask::didComplete(const ResourceError& error) -> ExceptionType
{
    ASSERT(RunLoop::isMain());

    if (m_stopped)
        return m_shouldSuppressTaskStoppedExceptions ? ExceptionType::None : ExceptionType::TaskAlreadyStopped;

    if (m_completed)
        return ExceptionType::CompleteAlreadyCalled;

    if (!m_responseSent && error.isNull())
        return ExceptionType::NoResponseSent;

    if (m_waitingForRedirectCompletionHandlerCallback && error.isNull())
        return ExceptionType::WaitingForRedirectCompletionHandler;

    m_completed = true;

    if (isSync()) {
        Vector<uint8_t> data = m_syncData.takeAsContiguous()->span();
        m_syncCompletionHandler(m_syncResponse, error, WTFMove(data));
    }

    m_process->send(Messages::WebPage::URLSchemeTaskDidComplete(m_urlSchemeHandler->identifier(), m_resourceLoaderID, error), *m_webPageID);
    m_urlSchemeHandler->taskCompleted(*pageProxyID(), *this);

    return ExceptionType::None;
}

void WebURLSchemeTask::pageDestroyed()
{
    ASSERT(RunLoop::isMain());

    m_pageProxyID = std::nullopt;
    m_webPageID = std::nullopt;
    m_process = nullptr;
    m_stopped = true;
    
    if (isSync()) {
        Locker locker { m_requestLock };
        m_syncCompletionHandler({ }, failedCustomProtocolSyncLoad(m_request), { });
    }
}

void WebURLSchemeTask::stop()
{
    ASSERT(RunLoop::isMain());
    ASSERT(!m_stopped);

    m_stopped = true;

    if (isSync()) {
        Locker locker { m_requestLock };
        m_syncCompletionHandler({ }, failedCustomProtocolSyncLoad(m_request), { });
    }
}

#if PLATFORM(COCOA)
NSURLRequest *WebURLSchemeTask::nsRequest() const
{
    Locker locker { m_requestLock };
    return m_request.nsURLRequest(WebCore::HTTPBodyUpdatePolicy::UpdateHTTPBody);
}
#endif

} // namespace WebKit
