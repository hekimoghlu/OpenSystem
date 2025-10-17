/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#include <wtf/WorkerPool.h>

namespace WTF {

class WorkerPool::Worker final : public AutomaticThread {
public:
    friend class WorkerPool;

    Worker(const AbstractLocker& locker, WorkerPool& pool, Box<Lock> lock, Ref<AutomaticThreadCondition>&& condition, Seconds timeout)
        : AutomaticThread(locker, lock, WTFMove(condition), timeout)
        , m_pool(pool)
    {
    }

    PollResult poll(const AbstractLocker&) final
    {
        if (m_pool.m_tasks.isEmpty())
            return PollResult::Wait;
        m_task = m_pool.m_tasks.takeFirst();
        if (!m_task)
            return PollResult::Stop;
        return PollResult::Work;
    }

    WorkResult work() final
    {
        m_task();
        m_task = nullptr;
        return WorkResult::Continue;
    }

    void threadDidStart() final
    {
        Locker locker { *m_pool.m_lock };
        m_pool.m_numberOfActiveWorkers++;
    }

    void threadIsStopping(const AbstractLocker&) final
    {
        m_pool.m_numberOfActiveWorkers--;
    }

    bool shouldSleep(const AbstractLocker& locker) final
    {
        return m_pool.shouldSleep(locker);
    }

    ASCIILiteral name() const final
    {
        return m_pool.name();
    }

private:
    WorkerPool& m_pool;
    Function<void()> m_task;
};

WorkerPool::WorkerPool(ASCIILiteral name, unsigned numberOfWorkers, Seconds timeout)
    : m_lock(Box<Lock>::create())
    , m_condition(AutomaticThreadCondition::create())
    , m_timeout(timeout)
    , m_name(name)
{
    Locker locker { *m_lock };
    for (unsigned i = 0; i < numberOfWorkers; ++i)
        m_workers.append(adoptRef(*new Worker(locker, *this, m_lock, m_condition.copyRef(), timeout)));
}

WorkerPool::~WorkerPool()
{
    {
        Locker locker { *m_lock };
        for (unsigned i = m_workers.size(); i--;)
            m_tasks.append(nullptr); // Use null task to indicate that we want the thread to terminate.
        m_condition->notifyAll(locker);
    }
    for (auto& worker : m_workers)
        worker->join();
    ASSERT(!m_numberOfActiveWorkers);
}

bool WorkerPool::shouldSleep(const AbstractLocker&)
{
    if (m_timeout > 0_s && m_timeout.isInfinity())
        return false;

    MonotonicTime currentTime = MonotonicTime::now();
    if (m_lastTimeoutTime.isNaN() || (currentTime >= (m_lastTimeoutTime  + m_timeout))) {
        m_lastTimeoutTime = currentTime;
        return true;
    }
    return false;
}

void WorkerPool::postTask(Function<void()>&& task)
{
    Locker locker { *m_lock };
    m_tasks.append(WTFMove(task));
    m_condition->notifyOne(locker);
}

}
