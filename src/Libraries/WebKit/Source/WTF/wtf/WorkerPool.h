/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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

#include <wtf/AutomaticThread.h>
#include <wtf/Deque.h>
#include <wtf/Function.h>
#include <wtf/NumberOfCores.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WTF {

class WorkerPool : public ThreadSafeRefCounted<WorkerPool> {
public:
    WTF_EXPORT_PRIVATE void postTask(Function<void()>&&);

    WTF_EXPORT_PRIVATE ~WorkerPool();

    // If timeout is infinity, it means AutomaticThread will be never automatically destroyed.
    static Ref<WorkerPool> create(ASCIILiteral name, unsigned numberOfWorkers  = WTF::numberOfProcessorCores(), Seconds timeout = Seconds::infinity())
    {
        ASSERT(numberOfWorkers >= 1);
        return adoptRef(*new WorkerPool(name, numberOfWorkers, timeout));
    }

    ASCIILiteral name() const { return m_name; }

private:
    class Worker;
    friend class Worker;

    WTF_EXPORT_PRIVATE WorkerPool(ASCIILiteral name, unsigned numberOfWorkers, Seconds timeout);

    bool shouldSleep(const AbstractLocker&);

    Box<Lock> m_lock;
    Ref<AutomaticThreadCondition> m_condition;
    Seconds m_timeout;
    MonotonicTime m_lastTimeoutTime { MonotonicTime::nan() };
    unsigned m_numberOfActiveWorkers { 0 };
    Vector<Ref<Worker>> m_workers;
    Deque<Function<void()>> m_tasks;
    ASCIILiteral m_name;
};

}

using WTF::WorkerPool;
