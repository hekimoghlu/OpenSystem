/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
#include "XMLHttpRequestProgressEventThrottle.h"

#include "ContextDestructionObserverInlines.h"
#include "EventLoop.h"
#include "EventNames.h"
#include "EventTarget.h"
#include "ScriptExecutionContext.h"
#include "XMLHttpRequest.h"
#include "XMLHttpRequestProgressEvent.h"

namespace WebCore {

const Seconds XMLHttpRequestProgressEventThrottle::minimumProgressEventDispatchingInterval { 50_ms }; // 50 ms per specification.

XMLHttpRequestProgressEventThrottle::XMLHttpRequestProgressEventThrottle(XMLHttpRequest& target)
    : m_target(target)
{
}

XMLHttpRequestProgressEventThrottle::~XMLHttpRequestProgressEventThrottle() = default;

void XMLHttpRequestProgressEventThrottle::updateProgress(bool isAsync, bool lengthComputable, unsigned long long loaded, unsigned long long total)
{
    m_lengthComputable = lengthComputable;
    m_loaded = loaded;
    m_total = total;

    if (!isAsync || !m_target.hasEventListeners(eventNames().progressEvent))
        return;

    if (!m_shouldDeferEventsDueToSuspension && !m_dispatchThrottledProgressEventTimer) {
        // The timer is not active so the least frequent event for now is every byte. Just dispatch the event.

        // We should not have any throttled progress event.
        ASSERT(!m_hasPendingThrottledProgressEvent);

        dispatchEventWhenPossible(XMLHttpRequestProgressEvent::create(eventNames().progressEvent, lengthComputable, loaded, total));
        m_dispatchThrottledProgressEventTimer = m_target.scriptExecutionContext()->eventLoop().scheduleRepeatingTask(
            minimumProgressEventDispatchingInterval, minimumProgressEventDispatchingInterval, TaskSource::Networking, [weakThis = WeakPtr { *this }] {
                if (weakThis)
                    weakThis->dispatchThrottledProgressEventTimerFired();
            });
        m_hasPendingThrottledProgressEvent = false;
        return;
    }

    // The timer is already active so minimumProgressEventDispatchingInterval is the least frequent event.
    m_hasPendingThrottledProgressEvent = true;
}

void XMLHttpRequestProgressEventThrottle::dispatchReadyStateChangeEvent(Event& event, ProgressEventAction progressEventAction)
{
    if (progressEventAction == FlushProgressEvent)
        flushProgressEvent();

    dispatchEventWhenPossible(event);
}

void XMLHttpRequestProgressEventThrottle::dispatchEventWhenPossible(Event& event)
{
    if (m_shouldDeferEventsDueToSuspension)
        m_target.queueTaskToDispatchEvent(m_target, TaskSource::Networking, event);
    else
        m_target.dispatchEvent(event);
}

void XMLHttpRequestProgressEventThrottle::dispatchProgressEvent(const AtomString& type)
{
    ASSERT(type == eventNames().loadstartEvent || type == eventNames().progressEvent || type == eventNames().loadEvent || type == eventNames().loadendEvent);

    if (type == eventNames().loadstartEvent) {
        m_lengthComputable = false;
        m_loaded = 0;
        m_total = 0;
    }

    if (m_target.hasEventListeners(type))
        dispatchEventWhenPossible(XMLHttpRequestProgressEvent::create(type, m_lengthComputable, m_loaded, m_total));
}

void XMLHttpRequestProgressEventThrottle::dispatchErrorProgressEvent(const AtomString& type)
{
    ASSERT(type == eventNames().loadendEvent || type == eventNames().abortEvent || type == eventNames().errorEvent || type == eventNames().timeoutEvent);

    if (m_target.hasEventListeners(type))
        dispatchEventWhenPossible(XMLHttpRequestProgressEvent::create(type, false, 0, 0));
}

void XMLHttpRequestProgressEventThrottle::flushProgressEvent()
{
    if (!m_hasPendingThrottledProgressEvent)
        return;

    m_hasPendingThrottledProgressEvent = false;
    // We stop the timer as this is called when no more events are supposed to occur.
    m_dispatchThrottledProgressEventTimer = nullptr;

    dispatchEventWhenPossible(XMLHttpRequestProgressEvent::create(eventNames().progressEvent, m_lengthComputable, m_loaded, m_total));
}

void XMLHttpRequestProgressEventThrottle::dispatchThrottledProgressEventTimerFired()
{
    ASSERT(m_dispatchThrottledProgressEventTimer);
    if (!m_hasPendingThrottledProgressEvent) {
        // No progress event was queued since the previous dispatch, we can safely stop the timer.
        m_dispatchThrottledProgressEventTimer = nullptr;
        return;
    }

    dispatchEventWhenPossible(XMLHttpRequestProgressEvent::create(eventNames().progressEvent, m_lengthComputable, m_loaded, m_total));
    m_hasPendingThrottledProgressEvent = false;
}

void XMLHttpRequestProgressEventThrottle::suspend()
{
    m_shouldDeferEventsDueToSuspension = true;

    if (m_hasPendingThrottledProgressEvent) {
        m_target.queueTaskKeepingObjectAlive(m_target, TaskSource::Networking, [this] {
            flushProgressEvent();
        });
    }
}

void XMLHttpRequestProgressEventThrottle::resume()
{
    m_target.queueTaskKeepingObjectAlive(m_target, TaskSource::Networking, [this] {
        m_shouldDeferEventsDueToSuspension = false;
    });
}

} // namespace WebCore
