/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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

#include "Decoder.h"
#include "Encoder.h"
#include <wtf/UniStdExtras.h>

#if OS(LINUX)
#include <poll.h>
#include <sys/eventfd.h>
#endif

namespace IPC {

Semaphore::Semaphore()
{
#if OS(LINUX)
    m_fd = { eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK | EFD_SEMAPHORE), UnixFileDescriptor::Adopt };
#endif
}

Semaphore::Semaphore(UnixFileDescriptor&& fd)
    : m_fd(WTFMove(fd))
{ }

Semaphore::Semaphore(Semaphore&&) = default;
Semaphore& Semaphore::operator=(Semaphore&&) = default;

Semaphore::~Semaphore()
{
    destroy();
}

void Semaphore::signal()
{
#if OS(LINUX)
    ASSERT_WITH_MESSAGE(m_fd.value() >= 0, "Signalling on an invalid semaphore object");

    // Matching waitImpl() and EFD_SEMAPHORE semantics, increment the semaphore value by 1.
    uint64_t value = 1;
    while (true) {
        int ret = write(m_fd.value(), &value, sizeof(uint64_t));
        if (LIKELY(ret != -1 || errno != EINTR))
            break;
    }
#endif
}

#if OS(LINUX)
static bool waitImpl(int fd, int timeout)
{
    struct pollfd pollfdValue { .fd = fd, .events = POLLIN, .revents = 0 };
    int ret = 0;

    // Iterate on polling while interrupts are thrown, otherwise bail out of the loop immediately.
    while (true) {
        ret = poll(&pollfdValue, 1, timeout);
        if (LIKELY(ret != -1 || errno != EINTR))
            break;
    }

    // Fail if the return value doesn't indicate the single file descriptor with only input events available.
    if (ret != 1 || !!(pollfdValue.revents ^ POLLIN))
        return false;

    // There should be data for reading -- due to EFD_SEMAPHORE, it should be 1 packed into an 8-byte value.
    uint64_t value = 0;
    ret = read(fd, &value, sizeof(uint64_t));
    if (ret != sizeof(uint64_t) || value != 1)
        return false;

    return true;
}
#endif

bool Semaphore::wait()
{
#if OS(LINUX)
    ASSERT_WITH_MESSAGE(m_fd.value() >= 0, "Waiting on an invalid semaphore object");

    return waitImpl(m_fd.value(), -1);
#else
    return false;
#endif
}

bool Semaphore::waitFor(Timeout timeout)
{
#if OS(LINUX)
    ASSERT_WITH_MESSAGE(m_fd.value() >= 0, "Waiting on an invalid semaphore object");

    int timeoutValue = -1;
    if (!timeout.isInfinity())
        timeoutValue = int(timeout.secondsUntilDeadline().milliseconds());
    return waitImpl(m_fd.value(), timeoutValue);
#else
    return false;
#endif
}

UnixFileDescriptor Semaphore::duplicateDescriptor() const
{
    return m_fd.duplicate();
}

void Semaphore::destroy()
{
    m_fd = { };
}

} // namespace IPC
