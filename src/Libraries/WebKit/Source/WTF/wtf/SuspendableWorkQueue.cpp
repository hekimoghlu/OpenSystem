/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#include <wtf/SuspendableWorkQueue.h>

#include <wtf/Logging.h>

namespace WTF {

Ref<SuspendableWorkQueue> SuspendableWorkQueue::create(ASCIILiteral name, WorkQueue::QOS qos, ShouldLog shouldLog)
{
    return adoptRef(*new SuspendableWorkQueue(name, qos, shouldLog));
}

SuspendableWorkQueue::SuspendableWorkQueue(ASCIILiteral name, QOS qos, ShouldLog shouldLog)
    : WorkQueue(name, qos)
    , m_shouldLog(shouldLog == ShouldLog::Yes)
{
    ASSERT(isMainThread());
}

ASCIILiteral SuspendableWorkQueue::stateString(State state)
{
    switch (state) {
    case State::Running:
        return "Running"_s;
    case State::WillSuspend:
        return "WillSuspend"_s;
    case State::Suspended:
        return "Suspended"_s;
    }

    ASSERT_NOT_REACHED();
    return { };
}

void SuspendableWorkQueue::suspend(Function<void()>&& suspendFunction, CompletionHandler<void()>&& completionHandler)
{
    ASSERT(isMainThread());
    Locker suspensionLocker { m_suspensionLock };

    RELEASE_LOG_IF(m_shouldLog, SuspendableWorkQueue, "%p - SuspendableWorkQueue::suspend current state %" PUBLIC_LOG_STRING, this, stateString(m_state).characters());
    if (m_state == State::Suspended)
        return completionHandler();

    // Last suspend function will be the one that is used.
    m_suspendFunction = WTFMove(suspendFunction);
    m_suspensionCompletionHandlers.append(WTFMove(completionHandler));
    if (m_state == State::WillSuspend)
        return;

    m_state = State::WillSuspend;
    // Make sure queue will be suspended when there is no task scheduled on the queue.
    WorkQueue::dispatch([this] {
        suspendIfNeeded();
    });
}

void SuspendableWorkQueue::resume()
{
    ASSERT(isMainThread());
    Locker suspensionLocker { m_suspensionLock };

    RELEASE_LOG_IF(m_shouldLog, SuspendableWorkQueue, "%p - SuspendableWorkQueue::resume current state %" PUBLIC_LOG_STRING, this, stateString(m_state).characters());
    if (m_state == State::Running)
        return;

    if (m_state == State::Suspended)
        m_suspensionCondition.notifyOne();

    m_state = State::Running;
}

void SuspendableWorkQueue::dispatch(Function<void()>&& function)
{
    RELEASE_ASSERT(function);
    // WorkQueue will protect this in dispatch().
    WorkQueue::dispatch([this, function = WTFMove(function)] {
        suspendIfNeeded();
        function();
    });
}

void SuspendableWorkQueue::dispatchAfter(Seconds seconds, Function<void()>&& function)
{
    RELEASE_ASSERT(function);
    WorkQueue::dispatchAfter(seconds, [this, function = WTFMove(function)] {
        suspendIfNeeded();
        function();
    });
}

void SuspendableWorkQueue::dispatchSync(Function<void()>&& function)
{
    // This function should be called only when queue is not about to be suspended,
    // otherwise thread may be blocked.
    if (isMainThread()) {
        Locker suspensionLocker { m_suspensionLock };
        RELEASE_ASSERT(m_state == State::Running);
    }
    WorkQueue::dispatchSync(WTFMove(function));
}

void SuspendableWorkQueue::invokeAllSuspensionCompletionHandlers()
{
    ASSERT(!isMainThread());

    if (m_suspensionCompletionHandlers.isEmpty())
        return;

    callOnMainThread([completionHandlers = std::exchange(m_suspensionCompletionHandlers, { })]() mutable {
        for (auto& completionHandler : completionHandlers) {
            if (completionHandler)
                completionHandler();
        }
    });
}

void SuspendableWorkQueue::suspendIfNeeded()
{
    ASSERT(!isMainThread());
    Locker suspensionLocker { m_suspensionLock };

    auto suspendFunction = std::exchange(m_suspendFunction, { });
    if (m_state != State::WillSuspend) {
        // If state is suspended, we should not reach here.
        RELEASE_LOG_ERROR_IF(m_shouldLog && m_state == State::Suspended, SuspendableWorkQueue, "%p - SuspendableWorkQueue::suspendIfNeeded current state Suspended", this);
        return;
    }

    RELEASE_LOG_IF(m_shouldLog, SuspendableWorkQueue, "%p - SuspendableWorkQueue::suspendIfNeeded set state to Suspended, will begin suspension", this);
    m_state = State::Suspended;
    suspendFunction();
    invokeAllSuspensionCompletionHandlers();

    while (m_state == State::Suspended)
        m_suspensionCondition.wait(m_suspensionLock);

    RELEASE_LOG_IF(m_shouldLog, SuspendableWorkQueue, "%p - SuspendableWorkQueue::suspendIfNeeded end suspension", this);
}

} // namespace WTF
