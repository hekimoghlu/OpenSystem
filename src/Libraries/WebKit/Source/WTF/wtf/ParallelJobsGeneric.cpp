/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#include <wtf/ParallelJobsGeneric.h>

#if ENABLE(THREADING_GENERIC)

#include <wtf/NumberOfCores.h>

namespace WTF {

Vector< RefPtr<ParallelEnvironment::ThreadPrivate> >* ParallelEnvironment::s_threadPool = nullptr;

ParallelEnvironment::ParallelEnvironment(ThreadFunction threadFunction, size_t sizeOfParameter, int requestedJobNumber) :
    m_threadFunction(threadFunction),
    m_sizeOfParameter(sizeOfParameter)
{
    ASSERT_ARG(requestedJobNumber, requestedJobNumber >= 1);

    int maxNumberOfCores = numberOfProcessorCores();

    if (!requestedJobNumber || requestedJobNumber > maxNumberOfCores)
        requestedJobNumber = static_cast<unsigned>(maxNumberOfCores);

    if (!s_threadPool)
        s_threadPool = new Vector< RefPtr<ThreadPrivate> >();

    // The main thread should be also a worker.
    int maxNumberOfNewThreads = requestedJobNumber - 1;

    for (int i = 0; i < maxNumberOfCores && m_threads.size() < static_cast<unsigned>(maxNumberOfNewThreads); ++i) {
        if (s_threadPool->size() < static_cast<unsigned>(i) + 1U)
            s_threadPool->append(ThreadPrivate::create());

        if ((*s_threadPool)[i]->tryLockFor(this))
            m_threads.append((*s_threadPool)[i]);
    }

    m_numberOfJobs = m_threads.size() + 1;
}

void ParallelEnvironment::execute(void* parameters)
{
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    unsigned char* currentParameter = static_cast<unsigned char*>(parameters);
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    size_t i;
    for (i = 0; i < m_threads.size(); ++i) {
        m_threads[i]->execute(m_threadFunction, currentParameter);
        currentParameter += m_sizeOfParameter;
    }

    // The work for the main thread.
    (*m_threadFunction)(currentParameter);

    // Wait until all jobs are done.
    for (i = 0; i < m_threads.size(); ++i)
        m_threads[i]->waitForFinish();
}

bool ParallelEnvironment::ThreadPrivate::tryLockFor(ParallelEnvironment* parent)
{
    bool locked = m_lock.tryLock();

    if (!locked)
        return false;

    if (m_parent) {
        m_lock.unlock();
        return false;
    }

    if (!m_thread) {
        m_thread = Thread::create("Parallel worker"_s, [this] {
            Locker lock { m_lock };

            while (true) {
                if (m_running) {
                    (*m_threadFunction)(m_parameters);
                    m_running = false;
                    m_parent = nullptr;
                    m_threadCondition.notifyOne();
                }

                m_threadCondition.wait(m_lock);
            }
        });
    }
    m_parent = parent;

    m_lock.unlock();
    return true;
}

void ParallelEnvironment::ThreadPrivate::execute(ThreadFunction threadFunction, void* parameters)
{
    Locker lock { m_lock };

    m_threadFunction = threadFunction;
    m_parameters = parameters;
    m_running = true;
    m_threadCondition.notifyOne();
}

void ParallelEnvironment::ThreadPrivate::waitForFinish()
{
    Locker lock { m_lock };

    while (m_running)
        m_threadCondition.wait(m_lock);
}

} // namespace WTF
#endif // ENABLE(THREADING_GENERIC)
