/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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

#include "EventLoop.h"
#include "GCReachableRef.h"
#include "Timer.h"
#include <wtf/HashSet.h>
#include <wtf/WeakHashMap.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CustomElementQueue;
class Document;
class HTMLSlotElement;
class MutationObserver;
class Page;
class SecurityOrigin;

// https://html.spec.whatwg.org/multipage/webappapis.html#window-event-loop
class WindowEventLoop final : public EventLoop {
public:
    static Ref<WindowEventLoop> eventLoopForSecurityOrigin(const SecurityOrigin&);

    virtual ~WindowEventLoop();

    void queueMutationObserverCompoundMicrotask();
    Vector<GCReachableRef<HTMLSlotElement>>& signalSlotList() { return m_signalSlotList; }
    UncheckedKeyHashSet<RefPtr<MutationObserver>>& activeMutationObservers() { return m_activeObservers; }
    UncheckedKeyHashSet<RefPtr<MutationObserver>>& suspendedMutationObservers() { return m_suspendedObservers; }

    CustomElementQueue& backupElementQueue();

    void scheduleIdlePeriod();
    void opportunisticallyRunIdleCallbacks(std::optional<MonotonicTime> deadline = std::nullopt);
    MonotonicTime computeIdleDeadline();

    WEBCORE_EXPORT static void breakToAllowRenderingUpdate();

private:
    static Ref<WindowEventLoop> create(const String&);
    WindowEventLoop(const String&);

    void scheduleToRun() final;
    bool isContextThread() const final;
    MicrotaskQueue& microtaskQueue() final;

    void startIdlePeriod(MonotonicTime);
    bool shouldEndIdlePeriod();
    std::optional<MonotonicTime> nextScheduledWorkTime() const;
    std::optional<MonotonicTime> nextRenderingTime() const;
    void didReachTimeToRun();
    void didFireIdleTimer();

    void decayIdleCallbackDuration() { m_expectedIdleCallbackDuration /= 2; }

    String m_agentClusterKey;
    Timer m_timer;
    Timer m_idleTimer;
    std::unique_ptr<MicrotaskQueue> m_microtaskQueue;

    // Each task scheduled in event loop is associated with a document so that it can be suspened or stopped
    // when the associated document is suspened or stopped. This task group is used to schedule a task
    // which is not scheduled to a specific document, and should only be used when it's absolutely required.
    EventLoopTaskGroup m_perpetualTaskGroupForSimilarOriginWindowAgents;

    bool m_mutationObserverCompoundMicrotaskQueuedFlag { false };
    bool m_deliveringMutationRecords { false }; // FIXME: This flag doesn't exist in the spec.
    Vector<GCReachableRef<HTMLSlotElement>> m_signalSlotList; // https://dom.spec.whatwg.org/#signal-slot-list
    UncheckedKeyHashSet<RefPtr<MutationObserver>> m_activeObservers;
    UncheckedKeyHashSet<RefPtr<MutationObserver>> m_suspendedObservers;

    std::unique_ptr<CustomElementQueue> m_customElementQueue;
    bool m_processingBackupElementQueue { false };

    MonotonicTime m_lastIdlePeriodStartTime;
    Seconds m_expectedIdleCallbackDuration { 4_ms };
};

} // namespace WebCore
