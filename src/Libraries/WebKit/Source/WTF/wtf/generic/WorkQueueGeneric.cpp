/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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

#include <wtf/threads/BinarySemaphore.h>

namespace WTF {

WorkQueueBase::WorkQueueBase(RunLoop& runLoop)
    : m_runLoop(&runLoop)
    , m_threadID(mainThreadID)
{
}

void WorkQueueBase::platformInitialize(ASCIILiteral name, Type, QOS qos)
{
    m_runLoop = RunLoop::create(name, ThreadType::Unknown, qos).ptr();
    BinarySemaphore semaphore;
    m_runLoop->dispatch([&] {
        m_threadID = Thread::current().uid();
        semaphore.signal();
    });
    semaphore.wait();
}

void WorkQueueBase::platformInvalidate()
{
    if (m_runLoop) {
        Ref<RunLoop> protector(*m_runLoop);
        protector->stop();
        protector->dispatch([] {
            RunLoop::current().stop();
        });
    }
}

void WorkQueueBase::dispatch(Function<void()>&& function)
{
    m_runLoop->dispatch([protectedThis = Ref { *this }, function = WTFMove(function)] {
        function();
    });
}

void WorkQueueBase::dispatchAfter(Seconds delay, Function<void()>&& function)
{
    m_runLoop->dispatchAfter(delay, [protectedThis = Ref { *this }, function = WTFMove(function)] {
        function();
    });
}

WorkQueue::WorkQueue(MainTag)
    : WorkQueueBase(RunLoop::main())
{
}

}
