/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
#ifndef _MACH_SEMAPHORE_H_
#define _MACH_SEMAPHORE_H_

#include <mach/port.h>
#include <mach/mach_types.h>
#include <mach/kern_return.h>
#include <mach/sync_policy.h>

/*
 *	Forward Declarations
 *
 *	The semaphore creation and deallocation routines are
 *	defined with the Mach task APIs in <mach/task.h>.
 *
 *      kern_return_t	semaphore_create(task_t task,
 *                                       semaphore_t *new_semaphore,
 *					 sync_policy_t policy,
 *					 int value);
 *
 *	kern_return_t	semaphore_destroy(task_t task,
 *					  semaphore_t semaphore);
 */

#include <sys/cdefs.h>
__BEGIN_DECLS

extern  kern_return_t   semaphore_signal(semaphore_t semaphore);
extern  kern_return_t   semaphore_signal_all(semaphore_t semaphore);

extern  kern_return_t   semaphore_wait(semaphore_t semaphore);

#ifdef  KERNEL

#ifdef  __LP64__

#ifdef  KERNEL_PRIVATE

extern  kern_return_t   semaphore_timedwait(semaphore_t semaphore,
    mach_timespec_t wait_time);

#endif  /* KERNEL_PRIVATE */

#else   /* __LP64__ */

extern  kern_return_t   semaphore_timedwait(semaphore_t semaphore,
    mach_timespec_t wait_time);

#endif  /* __LP64__ */

extern  kern_return_t   semaphore_wait_deadline(semaphore_t semaphore,
    uint64_t deadline);
extern  kern_return_t   semaphore_wait_noblock(semaphore_t semaphore);

#ifdef  XNU_KERNEL_PRIVATE

extern  kern_return_t   semaphore_wait_signal(semaphore_t wait_semaphore,
    semaphore_t signal_semaphore);

extern  kern_return_t   semaphore_timedwait_signal(semaphore_t wait_semaphore,
    semaphore_t signal_semaphore,
    mach_timespec_t wait_time);

extern  kern_return_t   semaphore_signal_thread(semaphore_t semaphore,
    thread_t thread);

#endif  /* XNU_KERNEL_PRIVATE */

#else   /* KERNEL */

extern  kern_return_t   semaphore_timedwait(semaphore_t semaphore,
    mach_timespec_t wait_time);

extern  kern_return_t   semaphore_timedwait_signal(semaphore_t wait_semaphore,
    semaphore_t signal_semaphore,
    mach_timespec_t wait_time);

extern  kern_return_t   semaphore_wait_signal(semaphore_t wait_semaphore,
    semaphore_t signal_semaphore);

extern  kern_return_t   semaphore_signal_thread(semaphore_t semaphore,
    thread_t thread);

#endif  /* KERNEL */

__END_DECLS

#ifdef  PRIVATE

#define SEMAPHORE_OPTION_NONE           0x00000000

#define SEMAPHORE_SIGNAL                0x00000001
#define SEMAPHORE_WAIT                  0x00000002
#define SEMAPHORE_WAIT_ON_SIGNAL        0x00000008

#define SEMAPHORE_SIGNAL_TIMEOUT        0x00000010
#define SEMAPHORE_SIGNAL_ALL            0x00000020
#define SEMAPHORE_SIGNAL_INTERRUPT      0x00000040      /* libmach implements */
#define SEMAPHORE_SIGNAL_PREPOST        0x00000080

#define SEMAPHORE_WAIT_TIMEOUT          0x00000100
#define SEMAPHORE_WAIT_INTERRUPT        0x00000400      /* libmach implements */

#define SEMAPHORE_TIMEOUT_NOBLOCK       0x00100000
#define SEMAPHORE_TIMEOUT_RELATIVE      0x00200000

#define SEMAPHORE_USE_SAVED_RESULT      0x01000000      /* internal use only */
#define SEMAPHORE_SIGNAL_RELEASE        0x02000000      /* internal use only */
#define SEMAPHORE_THREAD_HANDOFF        0x04000000

#endif  /* PRIVATE */

#endif  /* _MACH_SEMAPHORE_H_ */
