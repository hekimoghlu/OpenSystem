/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
#include <mach/boolean.h>
#include <mach/message.h>
#include <mach/kern_return.h>
#include <mach/mach_traps.h>
#include <mach/mach_types.h>
#include <mach/clock_types.h>

kern_return_t
semaphore_signal(mach_port_t signal_semaphore)
{
	return semaphore_signal_trap(signal_semaphore);
}

kern_return_t
semaphore_signal_all(mach_port_t signal_semaphore)
{
	return semaphore_signal_all_trap(signal_semaphore);
}

kern_return_t
semaphore_signal_thread(mach_port_t signal_semaphore, mach_port_t thread_act)
{
	return semaphore_signal_thread_trap(signal_semaphore, thread_act);
}

kern_return_t
semaphore_wait(mach_port_t wait_semaphore)
{
	return semaphore_wait_trap(wait_semaphore);
}

kern_return_t
semaphore_timedwait(mach_port_t wait_semaphore, mach_timespec_t wait_time)
{
	return semaphore_timedwait_trap(wait_semaphore,
	           wait_time.tv_sec,
	           wait_time.tv_nsec);
}

kern_return_t
semaphore_wait_signal(mach_port_t wait_semaphore, mach_port_t signal_semaphore)
{
	return semaphore_wait_signal_trap(wait_semaphore, signal_semaphore);
}

kern_return_t
semaphore_timedwait_signal(mach_port_t wait_semaphore,
    mach_port_t signal_semaphore,
    mach_timespec_t wait_time)
{
	return semaphore_timedwait_signal_trap(wait_semaphore,
	           signal_semaphore,
	           wait_time.tv_sec,
	           wait_time.tv_nsec);
}
