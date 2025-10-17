/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
#include <wtf/WorkQueue.h>

#include <mutex>
#include <wtf/Condition.h>
#include <wtf/Deque.h>
#include <wtf/Function.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/NumberOfCores.h>
#include <wtf/Ref.h>
#include <wtf/Threading.h>
#include <wtf/threads/BinarySemaphore.h>

namespace WTF {

WorkQueueBase::WorkQueueBase(ASCIILiteral name, Type type, QOS qos)
{
    platformInitialize(name, type, qos);
}

WorkQueueBase::~WorkQueueBase()
{
    platformInvalidate();
}

Ref<ConcurrentWorkQueue> ConcurrentWorkQueue::create(ASCIILiteral name, QOS qos)
{
    return adoptRef(*new ConcurrentWorkQueue(name, qos));
}

void ConcurrentWorkQueue::dispatch(Function<void()>&& function)
{
    WorkQueueBase::dispatch(WTFMove(function));
}

#if !PLATFORM(COCOA)
void WorkQueueBase::dispatchSync(Function<void()>&& function)
{
    BinarySemaphore semaphore;
    dispatch([&semaphore, function = WTFMove(function)]() mutable {
        function();
        semaphore.signal();
    });
    semaphore.wait();
}

void WorkQueueBase::dispatchWithQOS(Function<void()>&& function, QOS)
{
    dispatch(WTFMove(function));
}

void ConcurrentWorkQueue::apply(size_t iterations, WTF::Function<void(size_t index)>&& function)
{
    if (!iterations)
        return;

    if (iterations == 1) {
        function(0);
        return;
    }

    class ThreadPool {
    public:
        ThreadPool()
            // We don't need a thread for the current core.
            : m_workers(numberOfProcessorCores() - 1, [this](size_t) {
                return Thread::create("ThreadPool Worker"_s, [this] {
                    threadBody();
                });
            })
        {
        }

        size_t workerCount() const { return m_workers.size(); }

        void dispatch(const WTF::Function<void ()>* function)
        {
            Locker locker { m_lock };
            m_queue.append(function);
            m_condition.notifyOne();
        }

    private:
        NO_RETURN void threadBody()
        {
            while (true) {
                const WTF::Function<void ()>* function;

                {
                    Locker locker { m_lock };
                    m_condition.wait(m_lock, [this] {
                        assertIsHeld(m_lock);
                        return !m_queue.isEmpty();
                    });

                    function = m_queue.takeFirst();
                }

                (*function)();
            }
        }

        Lock m_lock;
        Condition m_condition;
        Deque<const Function<void()>*> m_queue WTF_GUARDED_BY_LOCK(m_lock);

        Vector<Ref<Thread>> m_workers;
    };

    static LazyNeverDestroyed<ThreadPool> threadPool;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        threadPool.construct();
    });

    // Cap the worker count to the number of iterations (excluding this thread)
    const size_t workerCount = std::min(iterations - 1, threadPool->workerCount());

    std::atomic<size_t> currentIndex(0);
    std::atomic<size_t> activeThreads(workerCount + 1);

    Condition condition;
    Lock lock;

    Function<void ()> applier = [&, function = WTFMove(function)] {
        size_t index;

        // Call the function for as long as there are iterations left.
        while ((index = currentIndex++) < iterations)
            function(index);

        // If there are no active threads left, signal the caller.
        if (!--activeThreads) {
            Locker locker { lock };
            condition.notifyOne();
        }
    };

    for (size_t i = 0; i < workerCount; ++i)
        threadPool->dispatch(&applier);
    applier();

    Locker locker { lock };
    condition.wait(lock, [&] { return !activeThreads; });
}
#endif

WorkQueue& WorkQueue::main()
{
    static NeverDestroyed<RefPtr<WorkQueue>> mainWorkQueue;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        WTF::initialize();
        mainWorkQueue.get() = adoptRef(*new WorkQueue(CreateMain));
    });
    return *mainWorkQueue.get();
}

Ref<WorkQueue> WorkQueue::create(ASCIILiteral name, QOS qos)
{
    return adoptRef(*new WorkQueue(name, qos));
}

WorkQueue::WorkQueue(ASCIILiteral name, QOS qos)
    : WorkQueueBase(name, Type::Serial, qos)
{
}

void WorkQueue::dispatch(Function<void()>&& function)
{
    WorkQueueBase::dispatch(WTFMove(function));
}

bool WorkQueue::isCurrent() const
{
    return currentSequence() == m_threadID;
}

ConcurrentWorkQueue::ConcurrentWorkQueue(ASCIILiteral name, QOS qos)
    : WorkQueueBase(name, Type::Concurrent, qos)
{
}

}
