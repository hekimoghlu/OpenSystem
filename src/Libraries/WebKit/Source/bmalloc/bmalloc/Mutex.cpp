/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
#include "Mutex.h"

#include "ScopeExit.h"
#if BOS(DARWIN)
#include <mach/mach_traps.h>
#include <mach/thread_switch.h>
#endif
#include <thread>

namespace bmalloc {

static inline void yield()
{
#if BOS(DARWIN)
    constexpr mach_msg_timeout_t timeoutInMS = 1;
    thread_switch(MACH_PORT_NULL, SWITCH_OPTION_DEPRESS, timeoutInMS);
#else
    sched_yield();
#endif
}

void Mutex::lockSlowCase()
{
    // The longest critical section in bmalloc is much shorter than the
    // time it takes to make a system call to yield to the OS scheduler.
    // So, we try again a lot before we yield.
    static constexpr size_t aLot = 256;
    
    if (!m_isSpinning.exchange(true)) {
        auto clear = makeScopeExit([&] { m_isSpinning.store(false); });

        for (size_t i = 0; i < aLot; ++i) {
            if (try_lock())
                return;
        }
    }

    // Avoid spinning pathologically.
    while (!try_lock())
        yield();
}

} // namespace bmalloc
