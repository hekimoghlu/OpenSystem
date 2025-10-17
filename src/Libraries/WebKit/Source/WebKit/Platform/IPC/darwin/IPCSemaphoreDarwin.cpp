/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#include "IPCSemaphore.h"

#include <mach/mach.h>

namespace IPC {

Semaphore::Semaphore()
{
    auto ret = semaphore_create(mach_task_self(), &m_semaphore, SYNC_POLICY_FIFO, 0);
    ASSERT_UNUSED(ret, ret == KERN_SUCCESS);
}

Semaphore::Semaphore(Semaphore&& other)
    : m_sendRight(WTFMove(other.m_sendRight))
    , m_semaphore(std::exchange(other.m_semaphore, SEMAPHORE_NULL))
{
}

Semaphore::Semaphore(MachSendRight&& sendRight)
    : m_sendRight(WTFMove(sendRight))
    , m_semaphore(m_sendRight.sendRight())
{
    ASSERT(m_sendRight);
}

Semaphore::~Semaphore()
{
    destroy();
}

Semaphore& Semaphore::operator=(Semaphore&& other)
{
    if (this != &other) {
        destroy();
        m_sendRight = WTFMove(other.m_sendRight);
        m_semaphore = std::exchange(other.m_semaphore, SEMAPHORE_NULL);
    }

    return *this;
}

void Semaphore::signal()
{
    auto ret = semaphore_signal(m_semaphore);
    ASSERT_UNUSED(ret, ret == KERN_SUCCESS || ret == KERN_TERMINATED);
}

bool Semaphore::wait()
{
    auto ret = semaphore_wait(m_semaphore);
    ASSERT(ret == KERN_SUCCESS || ret == KERN_TERMINATED);
    return ret == KERN_SUCCESS;
}

bool Semaphore::waitFor(Timeout timeout)
{
    Seconds waitTime = timeout.secondsUntilDeadline();
    auto seconds = waitTime.secondsAs<unsigned>();
    auto ret = semaphore_timedwait(m_semaphore, { seconds, static_cast<clock_res_t>(waitTime.nanosecondsAs<uint64_t>() - seconds * NSEC_PER_SEC) });
    ASSERT(ret == KERN_SUCCESS || ret == KERN_OPERATION_TIMED_OUT || ret == KERN_TERMINATED || ret == KERN_ABORTED);
    return ret == KERN_SUCCESS;
}

MachSendRight Semaphore::createSendRight() const
{
    return MachSendRight::create(m_semaphore);
}

void Semaphore::destroy()
{
    if (m_sendRight) {
        m_sendRight = MachSendRight();
        m_semaphore = SEMAPHORE_NULL;
        return;
    }
    if (m_semaphore == SEMAPHORE_NULL)
        return;
    auto ret = semaphore_destroy(mach_task_self(), m_semaphore);
    ASSERT_UNUSED(ret, ret == KERN_SUCCESS);
    m_semaphore = SEMAPHORE_NULL;
}

} // namespace IPC
