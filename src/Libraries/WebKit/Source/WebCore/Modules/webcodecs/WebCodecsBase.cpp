/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
#include "WebCodecsBase.h"

#if ENABLE(WEB_CODECS)

#include "Event.h"
#include "EventNames.h"
#include "WebCodecsControlMessage.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebCodecsBase);

WebCodecsBase::WebCodecsBase(ScriptExecutionContext& context)
    : ActiveDOMObject(&context)
{
}

WebCodecsBase::~WebCodecsBase() = default;

void WebCodecsBase::queueControlMessageAndProcess(WebCodecsControlMessage&& message)
{
    if (m_isMessageQueueBlocked) {
        m_controlMessageQueue.append(WTFMove(message));
        return;
    }
    m_controlMessageQueue.append(WTFMove(message));
    processControlMessageQueue();
}

void WebCodecsBase::queueCodecControlMessageAndProcess(WebCodecsControlMessage&& message)
{
    incrementCodecQueueSize();
    // message holds a strong ref to ourselves already.
    queueControlMessageAndProcess({ *this, [this, message = WTFMove(message)]() mutable {
        if (isCodecSaturated())
            return WebCodecsControlMessageOutcome::NotProcessed;
        decrementCodecQueueSizeAndScheduleDequeueEvent();
        return message();
    } });
}

void WebCodecsBase::scheduleDequeueEvent()
{
    if (m_dequeueEventScheduled)
        return;

    m_dequeueEventScheduled = true;
    queueTaskKeepingObjectAlive(*this, TaskSource::MediaElement, [this]() mutable {
        dispatchEvent(Event::create(eventNames().dequeueEvent, Event::CanBubble::No, Event::IsCancelable::No));
        m_dequeueEventScheduled = false;
    });
}

void WebCodecsBase::processControlMessageQueue()
{
    while (!m_isMessageQueueBlocked && !m_controlMessageQueue.isEmpty()) {
        auto& frontMessage = m_controlMessageQueue.first();
        auto outcome = frontMessage();
        if (outcome == WebCodecsControlMessageOutcome::NotProcessed)
            break;
        m_controlMessageQueue.removeFirst();
    }
}

void WebCodecsBase::incrementCodecQueueSize()
{
    m_codecControlMessagesPending++;
}

// Equivalent to spec's "Decrement [[encodeQueueSize]] or "Decrement [[decodeQueueSize]]" and run the Schedule Dequeue Event algorithm"
void WebCodecsBase::decrementCodecQueueSizeAndScheduleDequeueEvent()
{
    m_codecControlMessagesPending--;
    scheduleDequeueEvent();
}

void WebCodecsBase::decrementCodecOperationCountAndMaybeProcessControlMessageQueue()
{
    ASSERT(m_codecOperationsPending > 0);
    m_codecOperationsPending--;
    if (!isCodecSaturated())
        processControlMessageQueue();
}

void WebCodecsBase::clearControlMessageQueue()
{
    m_controlMessageQueue.clear();
}

void WebCodecsBase::clearControlMessageQueueAndMaybeScheduleDequeueEvent()
{
    clearControlMessageQueue();
    if (m_codecControlMessagesPending) {
        m_codecControlMessagesPending = 0;
        scheduleDequeueEvent();
    }
}

void WebCodecsBase::blockControlMessageQueue()
{
    m_isMessageQueueBlocked = true;
}

void WebCodecsBase::unblockControlMessageQueue()
{
    m_isMessageQueueBlocked = false;
    processControlMessageQueue();
}

bool WebCodecsBase::virtualHasPendingActivity() const
{
    return m_state == WebCodecsCodecState::Configured && (m_codecControlMessagesPending || m_isMessageQueueBlocked);
}

} // namespace WebCore

#endif // ENABLE(WEB_CODECS)
