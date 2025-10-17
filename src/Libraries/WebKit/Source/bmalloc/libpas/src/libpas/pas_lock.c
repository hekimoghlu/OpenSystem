/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_lock.h"
#if PAS_OS(DARWIN)
#include <mach/mach_traps.h>
#include <mach/thread_switch.h>
#endif

#if PAS_USE_SPINLOCKS

PAS_NEVER_INLINE void pas_lock_lock_slow(pas_lock* lock)
{
    static const size_t a_lot = 256;

    if (pas_compare_and_swap_bool_strong(&lock->is_spinning, false, true)) {
        size_t index;
        bool did_acquire;

        did_acquire = false;

        for (index = a_lot; index--;) {
            if (!pas_compare_and_swap_bool_strong(&lock->lock, false, true)) {
                did_acquire = true;
                break;
            }
        }

        lock->is_spinning = false;

        if (did_acquire)
            return;
    }

    while (!pas_compare_and_swap_bool_weak(&lock->lock, false, true)) {
#if PAS_OS(DARWIN)
        const mach_msg_timeout_t timeoutInMS = 1;
        thread_switch(MACH_PORT_NULL, SWITCH_OPTION_DEPRESS, timeoutInMS);
#else
        sched_yield();
#endif
    }
}

#endif /* PAS_USE_SPINLOCKS */

#endif /* LIBPAS_ENABLED */
