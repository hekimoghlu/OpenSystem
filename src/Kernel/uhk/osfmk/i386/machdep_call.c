/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
/*
 * Copyright (c) 1992 NeXT Computer, Inc.
 *
 * Machine dependent kernel calls.
 *
 * HISTORY
 *
 * 17 June 1992 ? at NeXT
 *	Created.
 */

#include <mach/mach_types.h>

#include <i386/machdep_call.h>

extern kern_return_t    kern_invalid(void);

const machdep_call_t            machdep_call_table[] = {
	MACHDEP_CALL_ROUTINE(kern_invalid, 0),
	MACHDEP_CALL_ROUTINE(kern_invalid, 0),
	MACHDEP_CALL_ROUTINE(kern_invalid, 0),
	MACHDEP_CALL_ROUTINE(thread_fast_set_cthread_self, 1),
	MACHDEP_CALL_ROUTINE(thread_set_user_ldt, 3),
	MACHDEP_BSD_CALL_ROUTINE(i386_set_ldt, 3),
	MACHDEP_BSD_CALL_ROUTINE(i386_get_ldt, 3),
};
const machdep_call_t            machdep_call_table64[] = {
#if HYPERVISOR
	MACHDEP_CALL_ROUTINE64(hv_task_trap, 2),
	MACHDEP_CALL_ROUTINE64(hv_thread_trap, 2),
#else
	MACHDEP_CALL_ROUTINE(kern_invalid, 0),
	MACHDEP_CALL_ROUTINE(kern_invalid, 0),
#endif
	MACHDEP_CALL_ROUTINE(kern_invalid, 0),
	MACHDEP_CALL_ROUTINE64(thread_fast_set_cthread_self64, 1),
	MACHDEP_CALL_ROUTINE(kern_invalid, 0),
	MACHDEP_BSD_CALL_ROUTINE64(i386_set_ldt64, 3),
	MACHDEP_BSD_CALL_ROUTINE64(i386_get_ldt64, 3)
};

int     machdep_call_count =
    (sizeof(machdep_call_table) / sizeof(machdep_call_t));
