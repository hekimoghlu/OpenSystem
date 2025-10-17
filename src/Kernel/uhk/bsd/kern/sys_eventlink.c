/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/kernel_types.h>
#include <sys/sysproto.h>
#include <mach/mach_types.h>
#include <mach/mach_eventlink_types.h>

extern uint64_t
mach_eventlink_signal_trap(
	mach_port_name_t port,
	uint64_t         signal_count __unused);

extern uint64_t
mach_eventlink_wait_until_trap(
	mach_port_name_t                    eventlink_port,
	uint64_t                            wait_count,
	mach_eventlink_signal_wait_option_t option,
	kern_clock_id_t                     clock_id,
	uint64_t                            deadline);

extern uint64_t
mach_eventlink_signal_wait_until_trap(
	mach_port_name_t                    eventlink_port,
	uint64_t                            wait_count,
	uint64_t                            signal_count __unused,
	mach_eventlink_signal_wait_option_t option,
	kern_clock_id_t                     clock_id,
	uint64_t                            deadline);

int
mach_eventlink_signal(
	__unused proc_t p,
	struct mach_eventlink_signal_args *uap,
	uint64_t *retval)
{
	*retval = mach_eventlink_signal_trap(uap->eventlink_port, uap->signal_count);
	return 0;
}

int
mach_eventlink_wait_until(
	__unused proc_t p,
	struct mach_eventlink_wait_until_args *uap,
	uint64_t *retval)
{
	*retval = mach_eventlink_wait_until_trap(uap->eventlink_port, uap->wait_count,
	    uap->option, uap->clock_id, uap->deadline);
	return 0;
}

int
mach_eventlink_signal_wait_until(
	__unused proc_t p,
	struct mach_eventlink_signal_wait_until_args *uap,
	uint64_t *retval)
{
	*retval = mach_eventlink_signal_wait_until_trap(uap->eventlink_port, uap->wait_count,
	    uap->signal_count, uap->option, uap->clock_id, uap->deadline);
	return 0;
}
