/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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
#ifndef _MACHDEP_INTERNAL_H_
#define _MACHDEP_INTERNAL_H_

#include <machine/types.h>

#include <pexpert/arm64/board_config.h>

#ifdef MACH_KERNEL_PRIVATE

/* We cache the following in EL0 registers
 *     TPIDRRO_EL0
 *         - the current cthread pointer
 *     TPIDR_EL0
 *         - the current CPU number (12 bits)
 *         - the current logical cluster id (8 bits)
 *
 * NOTE: Keep this in sync with libsyscall/os/tsd.h,
 *       specifically _os_cpu_number(), _os_cpu_cluster_number()
 */
#define MACHDEP_TPIDR_CPUNUM_SHIFT     0
#define MACHDEP_TPIDR_CPUNUM_MASK      0x0000000000000fff
#define MACHDEP_TPIDR_CLUSTERID_SHIFT  12
#define MACHDEP_TPIDR_CLUSTERID_MASK   0x00000000000ff000

#endif // MACH_KERNEL_PRIVATE

/*
 * Machine Thread Flags (machine_thread.flags)
 */

/* Thread is entitled to use x18, don't smash it when switching to thread. */
#if !__ARM_KERNEL_PROTECT__
#define ARM_MACHINE_THREAD_PRESERVE_X18_SHIFT           0
#define ARM_MACHINE_THREAD_PRESERVE_X18                 (1 << ARM_MACHINE_THREAD_PRESERVE_X18_SHIFT)
#endif /* !__ARM_KERNEL_PROTECT__ */

#if defined(HAS_APPLE_PAC)
#define ARM_MACHINE_THREAD_DISABLE_USER_JOP_SHIFT       1
#define ARM_MACHINE_THREAD_DISABLE_USER_JOP             (1 << ARM_MACHINE_THREAD_DISABLE_USER_JOP_SHIFT)
#endif /* HAS_APPLE_PAC */

/* Thread should use 1 GHz timebase */
#define ARM_MACHINE_THREAD_USES_1GHZ_TIMBASE_SHIFT      2
#define ARM_MACHINE_THREAD_USES_1GHZ_TIMBASE            (1 << ARM_MACHINE_THREAD_USES_1GHZ_TIMBASE_SHIFT)

#endif /* _MACHDEP_INTERNAL_H_ */
