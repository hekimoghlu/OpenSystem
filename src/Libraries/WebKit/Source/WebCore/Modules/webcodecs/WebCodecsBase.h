/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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

#if ENABLE(WEB_CODECS)

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "WebCodecsCodecState.h"
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

class WebCodecsControlMessage;

// WebCodecsBase implements the "Control Message Queue"
// as per https://w3c.github.io/webcodecs/#control-message-queue-slot
// And handle "Codec Saturation"
// as per https://w3c.github.io/webcodecs/#saturated
class WebCodecsBase
    : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<WebCodecsBase>
    , public ActiveDOMObject
    , public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebCodecsBase);
public:
    virtual ~WebCodecsBase();

    WebCodecsCodecState state() const { return m_state; }

    // ActiveDOMObject.
    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }
    bool virtualHasPendingActivity() const final;

protected:
    WebCodecsBase(ScriptExecutionContext&);
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }

    void setState(WebCodecsCodecState state) { m_state = state; }

    size_t codecQueueSize() const { return m_codecControlMessagesPending; }
    void queueControlMessageAndProcess(WebCodecsControlMessage&&);
    void queueCodecControlMessageAndProcess(WebCodecsControlMessage&&);
    void processControlMessageQueue();
    void clearControlMessageQueue();
    void clearControlMessageQueueAndMaybeScheduleDequeueEvent();
    void blockControlMessageQueue();
    void unblockControlMessageQueue();

    virtual size_t maximumCodecOperationsEnqueued() const { return 1; }
    void incrementCodecOperationCount() { m_codecOperationsPending++; };
    void decrementCodecOperationCountAndMaybeProcessControlMessageQueue();

private:
    // EventTarget
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // Equivalent to spec's "Increment [[encodeQueueSize]]." or "Increment [[decodeQueueSize]]"
    void incrementCodecQueueSize();
    // Equivalent to spec's "Decrement [[encodeQueueSize]] or "Decrement [[decodeQueueSize]]" and run the Schedule Dequeue Event algorithm"
    void decrementCodecQueueSizeAndScheduleDequeueEvent();
    bool isCodecSaturated() const { return m_codecOperationsPending >= maximumCodecOperationsEnqueued(); }
    void scheduleDequeueEvent();

    bool m_isMessageQueueBlocked { false };
    size_t m_codecControlMessagesPending { 0 };
    size_t m_codecOperationsPending { 0 };
    bool m_dequeueEventScheduled { false };
    Deque<WebCodecsControlMessage> m_controlMessageQueue;
    WebCodecsCodecState m_state { WebCodecsCodecState::Unconfigured };
};

} // namespace WebCore

#endif // ENABLE(WEB_CODECS)
