/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#ifndef _MACH_EVENTLINK_TYPES_H_
#define _MACH_EVENTLINK_TYPES_H_

#include <mach/std_types.h>
#include <mach/port.h>

__options_decl(kern_clock_id_t, uint32_t, {
	KERN_CLOCK_MACH_ABSOLUTE_TIME = 1,
});

__options_decl(mach_eventlink_create_option_t, uint32_t, {
	MELC_OPTION_NONE         = 0,
	MELC_OPTION_NO_COPYIN    = 0x1,
	MELC_OPTION_WITH_COPYIN  = 0x2,
});

__options_decl(mach_eventlink_associate_option_t, uint32_t, {
	MELA_OPTION_NONE              = 0,
	MELA_OPTION_ASSOCIATE_ON_WAIT = 0x1,
});

__options_decl(mach_eventlink_disassociate_option_t, uint32_t, {
	MELD_OPTION_NONE = 0,
});

__options_decl(mach_eventlink_signal_wait_option_t, uint32_t, {
	MELSW_OPTION_NONE    = 0,
	MELSW_OPTION_NO_WAIT = 0x1,
});

#define EVENTLINK_SIGNAL_COUNT_MASK 0xffffffffffffff
#define EVENTLINK_SIGNAL_ERROR_MASK 0xff
#define EVENTLINK_SIGNAL_ERROR_SHIFT 56

#define encode_eventlink_count_and_error(count, error) \
	(((count) & EVENTLINK_SIGNAL_COUNT_MASK) | ((((uint64_t)error) & EVENTLINK_SIGNAL_ERROR_MASK) << EVENTLINK_SIGNAL_ERROR_SHIFT))

#define decode_eventlink_count_from_retval(retval) \
	((retval) & EVENTLINK_SIGNAL_COUNT_MASK)

#define decode_eventlink_error_from_retval(retval) \
	((kern_return_t)(((retval) >> EVENTLINK_SIGNAL_ERROR_SHIFT) & EVENTLINK_SIGNAL_ERROR_MASK))

#ifndef KERNEL
kern_return_t
mach_eventlink_signal(
	mach_port_t         eventlink_port,
	uint64_t            signal_count);

kern_return_t
mach_eventlink_wait_until(
	mach_port_t                          eventlink_port,
	uint64_t                             *count_ptr,
	mach_eventlink_signal_wait_option_t  option,
	kern_clock_id_t                      clock_id,
	uint64_t                             deadline);

kern_return_t
mach_eventlink_signal_wait_until(
	mach_port_t                          eventlink_port,
	uint64_t                             *count_ptr,
	uint64_t                             signal_count,
	mach_eventlink_signal_wait_option_t  option,
	kern_clock_id_t                      clock_id,
	uint64_t                             deadline);

#endif

#endif  /* _MACH_EVENTLINK_TYPES_H_ */
