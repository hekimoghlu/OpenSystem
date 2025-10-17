/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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
#ifndef ParallelJobsGeneric_h
#define ParallelJobsGeneric_h

#if ENABLE(THREADING_GENERIC)

#include <wtf/Condition.h>
#include <wtf/Lock.h>
#include <wtf/RefCounted.h>
#include <wtf/Threading.h>

namespace WTF {

class ParallelEnvironment {
    WTF_MAKE_FAST_ALLOCATED;
public:
    typedef void (*ThreadFunction)(void*);

    WTF_EXPORT_PRIVATE ParallelEnvironment(ThreadFunction, size_t sizeOfParameter, int requestedJobNumber);

    int numberOfJobs()
    {
        return m_numberOfJobs;
    }

    WTF_EXPORT_PRIVATE void execute(void* parameters);

    class ThreadPrivate : public RefCounted<ThreadPrivate> {
    public:
        bool tryLockFor(ParallelEnvironment*);

        void execute(ThreadFunction, void*);

        void waitForFinish();

        static Ref<ThreadPrivate> create()
        {
            return adoptRef(*new ThreadPrivate());
        }

    private:
        mutable Lock m_lock;
        Condition m_threadCondition;

        RefPtr<Thread> m_thread;
        bool m_running { false };
        ParallelEnvironment* m_parent WTF_GUARDED_BY_LOCK(m_lock) { nullptr };

        ThreadFunction m_threadFunction { nullptr };
        void* m_parameters { nullptr };
    };

private:
    ThreadFunction m_threadFunction;
    size_t m_sizeOfParameter;
    int m_numberOfJobs;

    Vector< RefPtr<ThreadPrivate> > m_threads;
    static Vector< RefPtr<ThreadPrivate> >* s_threadPool;
};

} // namespace WTF

#endif // ENABLE(THREADING_GENERIC)


#endif // ParallelJobsGeneric_h
