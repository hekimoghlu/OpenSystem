/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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

#include <wtf/CompletionHandler.h>
#include <wtf/Condition.h>
#include <wtf/Deque.h>
#include <wtf/Lock.h>
#include <wtf/WorkQueue.h>

namespace WTF {

class WTF_EXPORT_PRIVATE SuspendableWorkQueue final : public WorkQueue {
public:
    using QOS = WorkQueue::QOS;
    enum class ShouldLog : bool { No, Yes };
    static Ref<SuspendableWorkQueue> create(ASCIILiteral name, QOS = QOS::Default, ShouldLog = ShouldLog::No);
    void suspend(Function<void()>&& suspendFunction, CompletionHandler<void()>&& suspensionCompletionHandler);
    void resume();
    void dispatch(Function<void()>&&) final;
    void dispatchAfter(Seconds, Function<void()>&&) final;
    void dispatchSync(Function<void()>&&) final;

private:
    SuspendableWorkQueue(ASCIILiteral name, QOS, ShouldLog);
    void invokeAllSuspensionCompletionHandlers() WTF_REQUIRES_LOCK(m_suspensionLock);
    void suspendIfNeeded();
#if USE(COCOA_EVENT_LOOP)
    using WorkQueue::dispatchQueue;
#else
    using WorkQueue::runLoop;
#endif
    enum class State : uint8_t { Running, WillSuspend, Suspended };
    static ASCIILiteral stateString(State);

    Lock m_suspensionLock;
    Condition m_suspensionCondition;
    State m_state WTF_GUARDED_BY_LOCK(m_suspensionLock) { State::Running };
    Function<void()> m_suspendFunction WTF_GUARDED_BY_LOCK(m_suspensionLock);
    Vector<CompletionHandler<void()>> m_suspensionCompletionHandlers WTF_GUARDED_BY_LOCK(m_suspensionLock);
    bool m_shouldLog WTF_GUARDED_BY_LOCK(m_suspensionLock) { false };
};

} // namespace WTF

using WTF::SuspendableWorkQueue;
