/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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

#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Page;
class WeakPtrImplWithEventTargetData;

template<typename T, typename WeakPtrImpl> class EventSender : public CanMakeCheckedPtr<EventSender<T, WeakPtrImpl>> {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(EventSender);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(EventSender);
    WTF_MAKE_NONCOPYABLE(EventSender);
public:
    EventSender();

    void dispatchEventSoon(T&, const AtomString& eventType);
    void cancelEvent(T&);
    void cancelEvent(T&, const AtomString& eventType);
    void dispatchPendingEvents(Page*);

#if ASSERT_ENABLED
    bool hasPendingEvents(T& sender) const
    {
        return m_dispatchSoonList.containsIf([&](auto& task) { return task.sender == &sender; })
            || m_dispatchingList.containsIf([&](auto& task) { return task.sender == &sender; });
    }
#endif

private:
    void timerFired() { dispatchPendingEvents(nullptr); }

    Timer m_timer;
    struct DispatchTask {
        WeakPtr<T, WeakPtrImpl> sender;
        AtomString eventType;
    };
    Vector<DispatchTask> m_dispatchSoonList;
    Vector<DispatchTask> m_dispatchingList;
};

#define TZONE_TEMPLATE_PARAMS template<typename T, typename WeakPtrImpl>
#define TZONE_TYPE EventSender<T, WeakPtrImpl>

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL_WITH_MULTIPLE_OR_SPECIALIZED_PARAMETERS();

#undef TZONE_TEMPLATE_PARAMS
#undef TZONE_TYPE


template<typename T, typename WeakPtrImpl> EventSender<T, WeakPtrImpl>::EventSender()
    : m_timer(*this, &EventSender::timerFired)
{
}

template<typename T, typename WeakPtrImpl> void EventSender<T, WeakPtrImpl>::dispatchEventSoon(T& sender, const AtomString& eventType)
{
    m_dispatchSoonList.append(DispatchTask { sender, eventType });
    if (!m_timer.isActive())
        m_timer.startOneShot(0_s);
}

template<typename T, typename WeakPtrImpl> void EventSender<T, WeakPtrImpl>::cancelEvent(T& senderToCancel)
{
    // Remove instances of this sender from both lists.
    // Use loops because we allow multiple instances to get into the lists.
    for (auto& [sender, eventType] : m_dispatchSoonList) {
        if (sender == &senderToCancel)
            sender = nullptr;
    }
    for (auto& [sender, eventType] : m_dispatchingList) {
        if (sender == &senderToCancel)
            sender = nullptr;
    }
}

template<typename T, typename WeakPtrImpl> void EventSender<T, WeakPtrImpl>::cancelEvent(T& senderToCancel, const AtomString& eventTypeToCancel)
{
    // Remove instances of this sender from both lists.
    // Use loops because we allow multiple instances to get into the lists.
    for (auto& [sender, eventType] : m_dispatchSoonList) {
        if (sender == &senderToCancel && eventType == eventTypeToCancel)
            sender = nullptr;
    }
    for (auto& [sender, eventType] : m_dispatchingList) {
        if (sender == &senderToCancel && eventType == eventTypeToCancel)
            sender = nullptr;
    }
}

template<typename T, typename WeakPtrImpl> void EventSender<T, WeakPtrImpl>::dispatchPendingEvents(Page* page)
{
    // Need to avoid re-entering this function; if new dispatches are
    // scheduled before the parent finishes processing the list, they
    // will set a timer and eventually be processed.
    if (!m_dispatchingList.isEmpty())
        return;

    m_timer.stop();

    m_dispatchSoonList.checkConsistency();

    m_dispatchingList = std::exchange(m_dispatchSoonList, { });
    for (auto& [weakSender, eventType] : m_dispatchingList) {
        if (!weakSender)
            continue;
        auto sender = std::exchange(weakSender, nullptr);
        if (!page || sender->document().page() == page)
            sender->dispatchPendingEvent(this, eventType);
        else
            dispatchEventSoon(*sender, eventType);
    }
    m_dispatchingList.clear();
}

} // namespace WebCore
