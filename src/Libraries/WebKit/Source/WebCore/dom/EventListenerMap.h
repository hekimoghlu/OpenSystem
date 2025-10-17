/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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

#include "RegisteredEventListener.h"
#include <atomic>
#include <memory>
#include <wtf/Forward.h>
#include <wtf/Lock.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class EventTarget;

using EventListenerVector = Vector<RefPtr<RegisteredEventListener>, 1, CrashOnOverflow, 2>;

class EventListenerMap {
public:
    WEBCORE_EXPORT EventListenerMap();

    bool isEmpty() const { return m_entries.isEmpty(); }
    bool contains(const AtomString& eventType) const { return find(eventType); }
    bool containsCapturing(const AtomString& eventType) const;
    bool containsActive(const AtomString& eventType) const;

    void clear();
    void clearEntriesForTearDown()
    {
        m_entries.clear();
    }

    void replace(const AtomString& eventType, EventListener& oldListener, Ref<EventListener>&& newListener, const RegisteredEventListener::Options&);
    bool add(const AtomString& eventType, Ref<EventListener>&&, const RegisteredEventListener::Options&);
    bool remove(const AtomString& eventType, EventListener&, bool useCapture);
    WEBCORE_EXPORT EventListenerVector* find(const AtomString& eventType);
    const EventListenerVector* find(const AtomString& eventType) const { return const_cast<EventListenerMap*>(this)->find(eventType); }
    Vector<AtomString> eventTypes() const;

    template<typename CallbackType>
    void enumerateEventListenerTypes(CallbackType callback) const
    {
        for (auto& entry : m_entries)
            callback(entry.first, entry.second.size());
    }

    template<typename CallbackType>
    bool containsMatchingEventListener(CallbackType callback) const
    {
        for (auto& entry : m_entries) {
            if (callback(entry.first, m_entries))
                return true;
        }
        return false;
    }

    void removeFirstEventListenerCreatedFromMarkup(const AtomString& eventType);
    void copyEventListenersNotCreatedFromMarkupToTarget(EventTarget*);
    
    template<typename Visitor> void visitJSEventListeners(Visitor&);
    Lock& lock() { return m_lock; }

private:
    Vector<std::pair<AtomString, EventListenerVector>, 0, CrashOnOverflow, 4> m_entries;
    Lock m_lock;
};

template<typename Visitor>
void EventListenerMap::visitJSEventListeners(Visitor& visitor)
{
    Locker locker { m_lock };
    for (auto& entry : m_entries) {
        for (auto& eventListener : entry.second)
            eventListener->callback().visitJSFunction(visitor);
    }
}

} // namespace WebCore
