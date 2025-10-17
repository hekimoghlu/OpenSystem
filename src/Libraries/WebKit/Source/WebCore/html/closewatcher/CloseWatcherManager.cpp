/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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
#include "CloseWatcherManager.h"

#include "Event.h"
#include "EventNames.h"
#include "KeyboardEvent.h"

namespace WebCore {

CloseWatcherManager::CloseWatcherManager() = default;

void CloseWatcherManager::add(Ref<CloseWatcher> watcher)
{
    if (m_groups.size() < m_allowedNumberOfGroups) {
        Vector<Ref<CloseWatcher>> newGroup;
        newGroup.append(watcher);
        m_groups.append(newGroup);
    } else {
        ASSERT(!m_groups.isEmpty());
        m_groups.last().append(watcher);
    }

    m_nextUserInteractionAllowsNewGroup = true;
}


void CloseWatcherManager::remove(CloseWatcher& watcher)
{
    for (auto& group : m_groups) {
        group.removeFirstMatching([&watcher] (const Ref<CloseWatcher>& current) {
            return current.ptr() == &watcher;
        });
        if (group.isEmpty())
            m_groups.removeFirst(group);
    }
}

void CloseWatcherManager::notifyAboutUserActivation()
{
    if (m_nextUserInteractionAllowsNewGroup)
        m_allowedNumberOfGroups++;

    m_nextUserInteractionAllowsNewGroup = false;
}

bool CloseWatcherManager::canPreventClose()
{
    return m_groups.size() < m_allowedNumberOfGroups;
}

void CloseWatcherManager::escapeKeyHandler(KeyboardEvent& event)
{
    if (!m_groups.isEmpty() && !event.defaultHandled() && event.isTrusted() && event.key() == "Escape"_s) {
        auto& group = m_groups.last();
        Vector<Ref<CloseWatcher>> groupCopy(group);
        for (Ref watcher : makeReversedRange(groupCopy)) {
            if (!watcher->requestToClose())
                break;
        }
    }

    if (m_allowedNumberOfGroups > 1)
        m_allowedNumberOfGroups--;
}

} // namespace WebCore
