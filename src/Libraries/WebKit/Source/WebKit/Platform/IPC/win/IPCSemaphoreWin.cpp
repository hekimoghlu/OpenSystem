/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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

#include <WebCore/SharedMemory.h>

namespace IPC {

Semaphore::Semaphore()
{
    m_semaphoreHandle = Win32Handle::adopt(::CreateSemaphoreA(nullptr, 0, 1, nullptr));
    RELEASE_ASSERT(m_semaphoreHandle);
}

Semaphore::Semaphore(Win32Handle&& handle)
    : m_semaphoreHandle(WTFMove(handle))
{
}

Semaphore::Semaphore(Semaphore&& other) = default;

Semaphore::~Semaphore()
{
    destroy();
}

Semaphore& Semaphore::operator=(Semaphore&& other)
{
    if (this != &other) {
        destroy();
        m_semaphoreHandle = WTFMove(other.m_semaphoreHandle);
    }

    return *this;
}

void Semaphore::signal()
{
    ReleaseSemaphore(m_semaphoreHandle.get(), 1, nullptr);
}

bool Semaphore::wait()
{
    return WAIT_OBJECT_0 == WaitForSingleObject(m_semaphoreHandle.get(), INFINITE);
}

bool Semaphore::waitFor(Timeout timeout)
{
    Seconds waitTime = timeout.secondsUntilDeadline();
    auto milliseconds = waitTime.millisecondsAs<DWORD>();
    return WAIT_OBJECT_0 == WaitForSingleObject(m_semaphoreHandle.get(), milliseconds);
}

void Semaphore::destroy()
{
    m_semaphoreHandle = { };
}

} // namespace IPC
