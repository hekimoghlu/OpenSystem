/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#include "WebURLSchemeHandler.h"

#include "URLSchemeTaskParameters.h"
#include "WebPageProxy.h"
#include "WebURLSchemeTask.h"

namespace WebKit {
using namespace WebCore;

WebURLSchemeHandler::WebURLSchemeHandler() = default;

WebURLSchemeHandler::~WebURLSchemeHandler()
{
    ASSERT(m_tasks.isEmpty());
}

void WebURLSchemeHandler::startTask(WebPageProxy& page, WebProcessProxy& process, PageIdentifier webPageID, URLSchemeTaskParameters&& parameters, SyncLoadCompletionHandler&& completionHandler)
{
    auto taskIdentifier = parameters.taskIdentifier;
    auto result = m_tasks.add({ taskIdentifier, page.identifier() }, WebURLSchemeTask::create(*this, page, process, webPageID, WTFMove(parameters), WTFMove(completionHandler)));
    ASSERT(result.isNewEntry);

    auto pageEntry = m_tasksByPageIdentifier.add(page.identifier(), HashSet<WebCore::ResourceLoaderIdentifier>());
    ASSERT(!pageEntry.iterator->value.contains(taskIdentifier));
    pageEntry.iterator->value.add(taskIdentifier);

    platformStartTask(page, result.iterator->value);
}

WebProcessProxy* WebURLSchemeHandler::processForTaskIdentifier(WebPageProxy& page, WebCore::ResourceLoaderIdentifier taskIdentifier) const
{
    auto key = std::make_pair(taskIdentifier, page.identifier());
    if (!decltype(m_tasks)::isValidKey(key))
        return nullptr;
    auto iterator = m_tasks.find(key);
    if (iterator == m_tasks.end())
        return nullptr;
    return iterator->value->process();
}

void WebURLSchemeHandler::stopAllTasksForPage(WebPageProxy& page, WebProcessProxy* process)
{
    auto iterator = m_tasksByPageIdentifier.find(page.identifier());
    if (iterator == m_tasksByPageIdentifier.end())
        return;

    auto& tasksByPage = iterator->value;
    auto taskIdentifiersToStop = WTF::compactMap(tasksByPage, [&](auto& taskIdentifier) -> std::optional<WebCore::ResourceLoaderIdentifier> {
        if (!process || processForTaskIdentifier(page, taskIdentifier) == process)
            return taskIdentifier;
        return std::nullopt;
    });

    for (auto& taskIdentifier : taskIdentifiersToStop)
        stopTask(page, taskIdentifier);

}

void WebURLSchemeHandler::stopTask(WebPageProxy& page, WebCore::ResourceLoaderIdentifier taskIdentifier)
{
    auto key = std::make_pair(taskIdentifier, page.identifier());
    if (!decltype(m_tasks)::isValidKey(key))
        return;
    auto iterator = m_tasks.find(key);
    if (iterator == m_tasks.end())
        return;

    iterator->value->stop();
    platformStopTask(page, iterator->value);

    removeTaskFromPageMap(page.identifier(), taskIdentifier);
    m_tasks.remove(iterator);
}

void WebURLSchemeHandler::taskCompleted(WebPageProxyIdentifier pageID, WebURLSchemeTask& task)
{
    auto takenTask = m_tasks.take({ task.resourceLoaderID(), pageID });
    ASSERT_UNUSED(takenTask, takenTask == &task);
    removeTaskFromPageMap(*task.pageProxyID(), task.resourceLoaderID());

    platformTaskCompleted(task);
}

void WebURLSchemeHandler::removeTaskFromPageMap(WebPageProxyIdentifier pageID, WebCore::ResourceLoaderIdentifier taskID)
{
    auto iterator = m_tasksByPageIdentifier.find(pageID);
    ASSERT(iterator != m_tasksByPageIdentifier.end());
    ASSERT(iterator->value.contains(taskID));
    if (!decltype(iterator->value)::isValidValue(taskID))
        return;
    iterator->value.remove(taskID);
    if (iterator->value.isEmpty())
        m_tasksByPageIdentifier.remove(iterator);
}

} // namespace WebKit
