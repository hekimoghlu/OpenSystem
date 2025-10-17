/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#include "EventListenerMap.h"

#include "AddEventListenerOptions.h"
#include "Event.h"
#include "EventTarget.h"
#include "JSEventListener.h"
#include <wtf/MainThread.h>
#include <wtf/StdLibExtras.h>
#include <wtf/Vector.h>


namespace WebCore {

EventListenerMap::EventListenerMap() = default;

bool EventListenerMap::containsCapturing(const AtomString& eventType) const
{
    auto* listeners = find(eventType);
    if (!listeners)
        return false;

    for (auto& eventListener : *listeners) {
        if (eventListener->useCapture())
            return true;
    }
    return false;
}

bool EventListenerMap::containsActive(const AtomString& eventType) const
{
    auto* listeners = find(eventType);
    if (!listeners)
        return false;

    for (auto& eventListener : *listeners) {
        if (!eventListener->isPassive())
            return true;
    }
    return false;
}

void EventListenerMap::clear()
{
    Locker locker { m_lock };

    for (auto& entry : m_entries) {
        for (auto& listener : entry.second)
            listener->markAsRemoved();
    }

    m_entries.clear();
}

Vector<AtomString> EventListenerMap::eventTypes() const
{
    return m_entries.map([](auto& entry) {
        return entry.first;
    });
}

static inline size_t findListener(const EventListenerVector& listeners, EventListener& listener, bool useCapture)
{
    for (size_t i = 0; i < listeners.size(); ++i) {
        auto& registeredListener = listeners[i];
        if (registeredListener->callback() == listener && registeredListener->useCapture() == useCapture)
            return i;
    }
    return notFound;
}

void EventListenerMap::replace(const AtomString& eventType, EventListener& oldListener, Ref<EventListener>&& newListener, const RegisteredEventListener::Options& options)
{
    Locker locker { m_lock };

    auto* listeners = find(eventType);
    ASSERT(listeners);
    size_t index = findListener(*listeners, oldListener, options.capture);
    ASSERT(index != notFound);
    auto& registeredListener = listeners->at(index);
    registeredListener->markAsRemoved();
    registeredListener = RegisteredEventListener::create(WTFMove(newListener), options);
}

bool EventListenerMap::add(const AtomString& eventType, Ref<EventListener>&& listener, const RegisteredEventListener::Options& options)
{
    Locker locker { m_lock };

    if (auto* listeners = find(eventType)) {
        if (findListener(*listeners, listener, options.capture) != notFound)
            return false; // Duplicate listener.
        listeners->append(RegisteredEventListener::create(WTFMove(listener), options));
        return true;
    }

    m_entries.append({ eventType, EventListenerVector { RegisteredEventListener::create(WTFMove(listener), options) } });
    return true;
}

static bool removeListenerFromVector(EventListenerVector& listeners, EventListener& listener, bool useCapture)
{
    size_t indexOfRemovedListener = findListener(listeners, listener, useCapture);
    if (UNLIKELY(indexOfRemovedListener == notFound))
        return false;

    listeners[indexOfRemovedListener]->markAsRemoved();
    listeners.remove(indexOfRemovedListener);
    return true;
}

bool EventListenerMap::remove(const AtomString& eventType, EventListener& listener, bool useCapture)
{
    Locker locker { m_lock };

    for (unsigned i = 0; i < m_entries.size(); ++i) {
        if (m_entries[i].first == eventType) {
            bool wasRemoved = removeListenerFromVector(m_entries[i].second, listener, useCapture);
            if (m_entries[i].second.isEmpty())
                m_entries.remove(i);
            return wasRemoved;
        }
    }

    return false;
}

EventListenerVector* EventListenerMap::find(const AtomString& eventType)
{
    for (auto& entry : m_entries) {
        if (entry.first == eventType)
            return &entry.second;
    }

    return nullptr;
}

static void removeFirstListenerCreatedFromMarkup(EventListenerVector& listenerVector)
{
    bool foundListener = listenerVector.removeFirstMatching([] (const auto& registeredListener) {
        if (JSEventListener::wasCreatedFromMarkup(registeredListener->callback())) {
            registeredListener->markAsRemoved();
            return true;
        }
        return false;
    });
    ASSERT_UNUSED(foundListener, foundListener);
}

void EventListenerMap::removeFirstEventListenerCreatedFromMarkup(const AtomString& eventType)
{
    Locker locker { m_lock };

    for (unsigned i = 0; i < m_entries.size(); ++i) {
        if (m_entries[i].first == eventType) {
            removeFirstListenerCreatedFromMarkup(m_entries[i].second);
            if (m_entries[i].second.isEmpty())
                m_entries.remove(i);
            return;
        }
    }
}

static void copyListenersNotCreatedFromMarkupToTarget(const AtomString& eventType, EventListenerVector& listenerVector, EventTarget* target)
{
    for (auto& registeredListener : listenerVector) {
        // Event listeners created from markup have already been transfered to the shadow tree during cloning.
        if (JSEventListener::wasCreatedFromMarkup(registeredListener->callback()))
            continue;
        target->addEventListener(eventType, registeredListener->callback(), registeredListener->useCapture());
    }
}

void EventListenerMap::copyEventListenersNotCreatedFromMarkupToTarget(EventTarget* target)
{
    for (auto& entry : m_entries)
        copyListenersNotCreatedFromMarkupToTarget(entry.first, entry.second, target);
}

} // namespace WebCore
