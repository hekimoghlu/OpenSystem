/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#include <wtf/Lock.h>

#include <wtf/LockAlgorithmInlines.h>
#include <wtf/StackShotProfiler.h>

#if OS(WINDOWS)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace WTF {

static constexpr bool profileLockContention = false;

void Lock::lockSlow()
{
    if (profileLockContention)
        STACK_SHOT_PROFILE(4, 2, 5);

    // Heap allocations are forbidden on certain threads (e.g. audio rendering thread) for performance reasons so we need to
    // explicitly allow the following allocation(s). In some rare cases, the lockSlow() algorithm may cause allocations.
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;

    DefaultLockAlgorithm::lockSlow(m_byte);
}

void Lock::unlockSlow()
{
    // Heap allocations are forbidden on certain threads (e.g. audio rendering thread) for performance reasons so we need to
    // explicitly allow the following allocation(s). In some rare cases, the unlockSlow() algorithm may cause allocations.
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;

    DefaultLockAlgorithm::unlockSlow(m_byte, DefaultLockAlgorithm::Unfair);
}

void Lock::unlockFairlySlow()
{
    // Heap allocations are forbidden on certain threads (e.g. audio rendering thread) for performance reasons so we need to
    // explicitly allow the following allocation(s). In some rare cases, the unlockSlow() algorithm may cause allocations.
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;

    DefaultLockAlgorithm::unlockSlow(m_byte, DefaultLockAlgorithm::Fair);
}

void Lock::safepointSlow()
{
    DefaultLockAlgorithm::safepointSlow(m_byte);
}

bool Lock::tryLockWithTimeout(Seconds timeout)
{
    // This function may be called from a signal handler (e.g. via visit()). Hence,
    // it should only use APIs that are safe to call from signal handlers. This is
    // why we use unistd.h's sleep() instead of its alternatives.

    // We'll be doing sleep(1) between tries below. Hence, sleepPerRetry is 1.
    unsigned maxRetries = (timeout < Seconds::infinity()) ? timeout.value() : std::numeric_limits<unsigned>::max();
    unsigned tryCount = 0;
    while (!tryLock() && tryCount++ <= maxRetries) {
#if OS(WINDOWS)
        Sleep(1000);
#else
        ::sleep(1);
#endif
    }
    return isHeld();
}

} // namespace WTF

