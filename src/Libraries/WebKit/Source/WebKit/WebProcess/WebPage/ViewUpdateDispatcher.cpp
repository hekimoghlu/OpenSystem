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
#include "config.h"
#include "ViewUpdateDispatcher.h"

#if ENABLE(UI_SIDE_COMPOSITING)

#include "Connection.h"
#include "ViewUpdateDispatcherMessages.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <WebCore/PageIdentifier.h>
#include <wtf/RunLoop.h>
#include <wtf/WorkQueue.h>

namespace WebKit {

ViewUpdateDispatcher::ViewUpdateDispatcher(WebProcess& process)
    : m_process(process)
    , m_queue(WorkQueue::create("com.apple.WebKit.ViewUpdateDispatcher"_s))
{
}

ViewUpdateDispatcher::~ViewUpdateDispatcher()
{
    ASSERT_NOT_REACHED();
}

void ViewUpdateDispatcher::ref() const
{
    m_process->ref();
}

void ViewUpdateDispatcher::deref() const
{
    m_process->deref();
}

void ViewUpdateDispatcher::initializeConnection(IPC::Connection& connection)
{
    connection.addMessageReceiver(m_queue.get(), *this, Messages::ViewUpdateDispatcher::messageReceiverName());
}

void ViewUpdateDispatcher::visibleContentRectUpdate(WebCore::PageIdentifier pageID, const VisibleContentRectUpdateInfo& visibleContentRectUpdateInfo)
{
    bool updateListWasEmpty;
    {
        Locker locker { m_latestUpdateLock };
        updateListWasEmpty = m_latestUpdate.isEmpty();
        auto iterator = m_latestUpdate.find(pageID);
        if (iterator == m_latestUpdate.end())
            m_latestUpdate.set(pageID, makeUniqueRef<UpdateData>(visibleContentRectUpdateInfo, visibleContentRectUpdateInfo.timestamp()));
        else
            iterator->value.get().visibleContentRectUpdateInfo = visibleContentRectUpdateInfo;
    }
    if (updateListWasEmpty) {
        RunLoop::main().dispatch([this] {
            dispatchVisibleContentRectUpdate();
        });
    }
}

void ViewUpdateDispatcher::dispatchVisibleContentRectUpdate()
{
    HashMap<WebCore::PageIdentifier, UniqueRef<UpdateData>> update;
    {
        Locker locker { m_latestUpdateLock };
        update = std::exchange(m_latestUpdate, { });
    }

    for (auto& slot : update) {
        if (WebPage* webPage = WebProcess::singleton().webPage(slot.key))
            webPage->updateVisibleContentRects(slot.value.get().visibleContentRectUpdateInfo, slot.value.get().oldestTimestamp);
    }
}

} // namespace WebKit

#endif // ENABLE(UI_SIDE_COMPOSITING)
