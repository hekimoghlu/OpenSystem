/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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

#include "RunLoopObserver.h"
#include <JavaScriptCore/EdenGCActivityCallback.h>
#include <JavaScriptCore/FullGCActivityCallback.h>
#include <JavaScriptCore/MarkedSpace.h>
#include <wtf/CheckedPtr.h>
#include <wtf/MonotonicTime.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class OpportunisticTaskScheduler;
class Page;

class ImminentlyScheduledWorkScope : public RefCounted<ImminentlyScheduledWorkScope> {
public:
    static Ref<ImminentlyScheduledWorkScope> create(OpportunisticTaskScheduler& scheduler)
    {
        return adoptRef(*new ImminentlyScheduledWorkScope(scheduler));
    }

    ~ImminentlyScheduledWorkScope();

private:
    ImminentlyScheduledWorkScope(OpportunisticTaskScheduler&);

    WeakPtr<OpportunisticTaskScheduler> m_scheduler;
};

class OpportunisticTaskScheduler final : public RefCountedAndCanMakeWeakPtr<OpportunisticTaskScheduler> {
public:
    static Ref<OpportunisticTaskScheduler> create(Page& page)
    {
        return adoptRef(*new OpportunisticTaskScheduler(page));
    }

    ~OpportunisticTaskScheduler();

    bool isScheduled() const { return m_runLoopObserver->isScheduled(); }
    void rescheduleIfNeeded(MonotonicTime deadline);
    bool hasImminentlyScheduledWork() const { return m_imminentlyScheduledWorkCount; }

    WARN_UNUSED_RETURN Ref<ImminentlyScheduledWorkScope> makeScheduledWorkScope();

    class FullGCActivityCallback final : public JSC::FullGCActivityCallback {
    public:
        using Base = JSC::FullGCActivityCallback;

        static Ref<FullGCActivityCallback> create(JSC::Heap& heap)
        {
            return adoptRef(*new FullGCActivityCallback(heap));
        }

        void doCollection(JSC::VM&) final;

    private:
        FullGCActivityCallback(JSC::Heap&);

        JSC::VM& m_vm;
        std::unique_ptr<RunLoopObserver> m_runLoopObserver;
        JSC::HeapVersion m_version { 0 };
        unsigned m_deferCount { 0 };
    };

    class EdenGCActivityCallback final : public JSC::EdenGCActivityCallback {
    public:
        using Base = JSC::EdenGCActivityCallback;

        static Ref<EdenGCActivityCallback> create(JSC::Heap& heap)
        {
            return adoptRef(*new EdenGCActivityCallback(heap));
        }

        void doCollection(JSC::VM&) final;

    private:
        EdenGCActivityCallback(JSC::Heap&);

        JSC::VM& m_vm;
        std::unique_ptr<RunLoopObserver> m_runLoopObserver;
        JSC::HeapVersion m_version { 0 };
        unsigned m_deferCount { 0 };
    };

private:
    friend class ImminentlyScheduledWorkScope;

    OpportunisticTaskScheduler(Page&);
    void runLoopObserverFired();

    bool isPageInactiveOrLoading() const;

    bool shouldAllowOpportunisticallyScheduledTasks() const;

    WeakPtr<Page> m_page;
    uint64_t m_imminentlyScheduledWorkCount { 0 };
    uint64_t m_runloopCountAfterBeingScheduled { 0 };
    MonotonicTime m_currentDeadline;
    std::unique_ptr<RunLoopObserver> m_runLoopObserver;
};

} // namespace WebCore
