/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
#include "WasmWorklist.h"
#include "WasmLLIntGenerator.h"

#if ENABLE(WEBASSEMBLY)

#include "CPU.h"
#include "WasmPlan.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace Wasm {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Worklist);

namespace WasmWorklistInternal {
static constexpr bool verbose = false;
}

void Worklist::dump(PrintStream& out) const
{
    out.print("Queue Size = ", m_queue.size());
}

// The Thread class is designed to prevent threads from blocking when there is still work
// in the queue. Wasm's Plans have some phases, Validiation, Preparation, and Completion,
// that can only be done by one thread, and another phase, Compilation, that can be done
// many threads. In order to stop a thread from wasting time we remove any plan that is
// is currently in a single threaded state from the work queue so other plans can run.
class Worklist::Thread final : public AutomaticThread {
public:
    using Base = AutomaticThread;
    static Ref<Thread> create(const AbstractLocker& locker, Worklist& work)
    {
        return adoptRef(*new Thread(locker, work));
    }

private:
    Thread(const AbstractLocker& locker, Worklist& work)
        : Base(locker, work.m_lock, work.m_planEnqueued.copyRef(), ThreadType::Compiler)
        , worklist(work)
    {

    }

    PollResult poll(const AbstractLocker&) final
    {
        auto& queue = worklist.m_queue;
        synchronize.notifyAll();

        while (!queue.isEmpty()) {
            Priority priority = queue.peek().priority;
            if (priority == Worklist::Priority::Shutdown)
                return PollResult::Stop;

            element = queue.peek();
            // Only one thread should validate/prepare.
            if (!queue.peek().plan->multiThreaded())
                queue.dequeue();

            if (element.plan->hasWork())
                return PollResult::Work;

            // There must be a another thread linking this plan so we can deque and see if there is other work.
            queue.dequeue();
            element = QueueElement();
        }
        return PollResult::Wait;
    }

    WorkResult work() final
    {
        auto complete = [&] (const AbstractLocker&) {
            // We need to hold the lock to release our plan otherwise the main thread, while canceling plans
            // might use after free the plan we are looking at
            element = QueueElement();
            return WorkResult::Continue;
        };

        Plan* plan = element.plan.get();
        ASSERT(plan);

        bool wasMultiThreaded = plan->multiThreaded();
        plan->work(Plan::Partial);

        ASSERT(!plan->hasWork() || plan->multiThreaded());
        if (plan->hasWork() && !wasMultiThreaded && plan->multiThreaded()) {
            Locker locker { *worklist.m_lock };
            element.setToNextPriority();
            worklist.m_queue.enqueue(WTFMove(element));
            worklist.m_planEnqueued->notifyAll(locker);
            return complete(locker);
        }

        return complete(Locker { *worklist.m_lock });
    }

    ASCIILiteral name() const final
    {
        return "Wasm Worklist Helper Thread"_s;
    }

public:
    Condition synchronize;
    Worklist& worklist;
    // We can only modify element when holding the lock. A synchronous compile might look at each thread's tasks in order to boost the priority.
    QueueElement element;
};

void Worklist::QueueElement::setToNextPriority()
{
    switch (priority) {
    case Priority::Preparation:
        priority = Priority::Compilation;
        return;
    case Priority::Synchronous:
        return;
    default:
        break;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void Worklist::enqueue(Ref<Plan> plan)
{
    Locker locker { *m_lock };

    if (ASSERT_ENABLED) {
        for (const auto& element : m_queue)
            ASSERT_UNUSED(element, element.plan.get() != &plan.get());
    }

    dataLogLnIf(WasmWorklistInternal::verbose, "Enqueuing plan");
    bool multiThreaded = plan->multiThreaded();
    m_queue.enqueue({ multiThreaded ? Priority::Compilation : Priority::Preparation, nextTicket(),  WTFMove(plan) });
    if (multiThreaded)
        m_planEnqueued->notifyAll(locker);
    else
        m_planEnqueued->notifyOne(locker);
}

void Worklist::completePlanSynchronously(Plan& plan)
{
    {
        Locker locker { *m_lock };
        m_queue.decreaseKey([&] (QueueElement& element) {
            if (element.plan == &plan) {
                element.priority = Priority::Synchronous;
                return true;
            }
            return false;
        });

        for (auto& thread : m_threads) {
            if (thread->element.plan == &plan)
                thread->element.priority = Priority::Synchronous;
        }
    }

    plan.waitForCompletion();
}

void Worklist::stopAllPlansForContext(VM& vm)
{
    Locker locker { *m_lock };
    Vector<QueueElement> elements;
    while (!m_queue.isEmpty()) {
        QueueElement element = m_queue.dequeue();
        bool didCancel = element.plan->tryRemoveContextAndCancelIfLast(vm);
        if (!didCancel)
            elements.append(WTFMove(element));
    }

    for (auto& element : elements)
        m_queue.enqueue(WTFMove(element));

    for (auto& thread : m_threads) {
        if (thread->element.plan) {
            bool didCancel = thread->element.plan->tryRemoveContextAndCancelIfLast(vm);
            if (didCancel) {
                // We don't have to worry about the deadlocking since the thread can't block without checking for a new plan and must hold the lock to do so.
                thread->synchronize.wait(*m_lock);
            }
        }
    }
}

Worklist::Worklist()
    : m_lock(Box<Lock>::create())
    , m_planEnqueued(AutomaticThreadCondition::create())
{
    unsigned numberOfCompilationThreads = Options::useConcurrentJIT() ? Options::numberOfWasmCompilerThreads() : 1;
    Locker locker { *m_lock };
    m_threads = Vector<Ref<Thread>>(numberOfCompilationThreads, [&](size_t) {
        return Worklist::Thread::create(locker, *this);
    });
}

Worklist::~Worklist()
{
    {
        Locker locker { *m_lock };
        m_queue.enqueue({ Priority::Shutdown, nextTicket(), nullptr });
        m_planEnqueued->notifyAll(locker);
    }
    for (unsigned i = 0; i < m_threads.size(); ++i)
        m_threads[i]->join();
}

static Worklist* globalWorklist;

Worklist* existingWorklistOrNull() { return globalWorklist; }
Worklist& ensureWorklist()
{
    static std::once_flag initializeWorklist;
    std::call_once(initializeWorklist, [] {
        Worklist* worklist = new Worklist();
        WTF::storeStoreFence();
        globalWorklist = worklist;
    });
    return *globalWorklist;
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
