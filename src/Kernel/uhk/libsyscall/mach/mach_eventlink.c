/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#include <mach/mach_eventlink_types.h>

/*
 * __mach_eventlink* calls are bsd syscalls instead of mach traps because
 * they need to return a 64 bit value in register and mach traps currently
 * does not allow 64 bit return values.
 */
uint64_t
__mach_eventlink_signal(
	mach_port_t         eventlink_port,
	uint64_t            signal_count);

uint64_t
__mach_eventlink_wait_until(
	mach_port_t                          eventlink_port,
	uint64_t                             wait_signal_count,
	uint64_t                             deadline,
	kern_clock_id_t                      clock_id,
	mach_eventlink_signal_wait_option_t  option);

uint64_t
__mach_eventlink_signal_wait_until(
	mach_port_t                          eventlink_port,
	uint64_t                             wait_count,
	uint64_t                             signal_count,
	uint64_t                             deadline,
	kern_clock_id_t                      clock_id,
	mach_eventlink_signal_wait_option_t  option);

kern_return_t
mach_eventlink_signal(
	mach_port_t         eventlink_port,
	uint64_t            signal_count)
{
	uint64_t retval = __mach_eventlink_signal(eventlink_port, signal_count);

	return decode_eventlink_error_from_retval(retval);
}

kern_return_t
mach_eventlink_wait_until(
	mach_port_t                          eventlink_port,
	uint64_t                             *wait_count_ptr,
	mach_eventlink_signal_wait_option_t  option,
	kern_clock_id_t                      clock_id,
	uint64_t                             deadline)
{
	uint64_t retval;

	retval = __mach_eventlink_wait_until(eventlink_port, *wait_count_ptr,
	    deadline, clock_id, option);

	*wait_count_ptr = decode_eventlink_count_from_retval(retval);
	return decode_eventlink_error_from_retval(retval);
}

kern_return_t
mach_eventlink_signal_wait_until(
	mach_port_t                          eventlink_port,
	uint64_t                             *wait_count_ptr,
	uint64_t                             signal_count,
	mach_eventlink_signal_wait_option_t  option,
	kern_clock_id_t                      clock_id,
	uint64_t                             deadline)
{
	uint64_t retval;
	retval = __mach_eventlink_signal_wait_until(eventlink_port, *wait_count_ptr,
	    signal_count, deadline, clock_id, option);
	*wait_count_ptr = decode_eventlink_count_from_retval(retval);
	return decode_eventlink_error_from_retval(retval);
}
