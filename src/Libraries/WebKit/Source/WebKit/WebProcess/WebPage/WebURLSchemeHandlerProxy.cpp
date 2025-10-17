/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
#include "WebURLSchemeHandlerProxy.h"

#include "MessageSenderInlines.h"
#include "URLSchemeTaskParameters.h"
#include "WebErrors.h"
#include "WebFrame.h"
#include "WebLoaderStrategy.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceLoader.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>

namespace WebKit {
using namespace WebCore;

WebURLSchemeHandlerProxy::WebURLSchemeHandlerProxy(WebPage& page, WebURLSchemeHandlerIdentifier identifier)
    : m_webPage(page)
    , m_identifier(identifier)
{
}

WebURLSchemeHandlerProxy::~WebURLSchemeHandlerProxy()
{
    ASSERT(m_tasks.isEmpty());
}

void WebURLSchemeHandlerProxy::startNewTask(ResourceLoader& loader, WebFrame& webFrame)
{
    auto result = m_tasks.add(*loader.identifier(), WebURLSchemeTaskProxy::create(*this, loader, webFrame));
    ASSERT(result.isNewEntry);

    WebProcess::singleton().webLoaderStrategy().addURLSchemeTaskProxy(*result.iterator->value);
    result.iterator->value->startLoading();
}

Ref<WebPage> WebURLSchemeHandlerProxy::protectedPage()
{
    return m_webPage.get();
}

void WebURLSchemeHandlerProxy::loadSynchronously(WebCore::ResourceLoaderIdentifier loadIdentifier, WebFrame& webFrame, const ResourceRequest& request, ResourceResponse& response, ResourceError& error, Vector<uint8_t>& data)
{
    data.shrink(0);
    auto sendResult = protectedPage()->sendSync(Messages::WebPageProxy::LoadSynchronousURLSchemeTask(URLSchemeTaskParameters { m_identifier, loadIdentifier, request, webFrame.info() }));
    if (sendResult.succeeded())
        std::tie(response, error, data) = sendResult.takeReply();
    else
        error = failedCustomProtocolSyncLoad(request);
}

void WebURLSchemeHandlerProxy::stopAllTasks()
{
    while (!m_tasks.isEmpty())
        m_tasks.begin()->value->stopLoading();
}

void WebURLSchemeHandlerProxy::taskDidPerformRedirection(WebCore::ResourceLoaderIdentifier taskIdentifier, WebCore::ResourceResponse&& redirectResponse, WebCore::ResourceRequest&& newRequest, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler)
{
    auto* task = m_tasks.get(taskIdentifier);
    if (!task)
        return;
    
    task->didPerformRedirection(WTFMove(redirectResponse), WTFMove(newRequest), WTFMove(completionHandler));
}

void WebURLSchemeHandlerProxy::taskDidReceiveResponse(WebCore::ResourceLoaderIdentifier taskIdentifier, const ResourceResponse& response)
{
    auto* task = m_tasks.get(taskIdentifier);
    if (!task)
        return;

    task->didReceiveResponse(response);
}

void WebURLSchemeHandlerProxy::taskDidReceiveData(WebCore::ResourceLoaderIdentifier taskIdentifier, Ref<WebCore::SharedBuffer>&& data)
{
    auto* task = m_tasks.get(taskIdentifier);
    if (!task)
        return;

    task->didReceiveData(WTFMove(data));
}

void WebURLSchemeHandlerProxy::taskDidComplete(WebCore::ResourceLoaderIdentifier taskIdentifier, const ResourceError& error)
{
    if (auto task = removeTask(taskIdentifier))
        task->didComplete(error);
}

void WebURLSchemeHandlerProxy::taskDidStopLoading(WebURLSchemeTaskProxy& task)
{
    ASSERT(m_tasks.get(task.identifier()) == &task);
    removeTask(task.identifier());
}

RefPtr<WebURLSchemeTaskProxy> WebURLSchemeHandlerProxy::removeTask(WebCore::ResourceLoaderIdentifier identifier)
{
    auto task = m_tasks.take(identifier);
    if (!task)
        return nullptr;

    WebProcess::singleton().webLoaderStrategy().removeURLSchemeTaskProxy(*task);

    return task;
}

} // namespace WebKit
