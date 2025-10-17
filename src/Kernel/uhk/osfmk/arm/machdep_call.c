/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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

#include <kern/thread.h>
#include <mach/mach_types.h>
#include <arm/machdep_call.h>
#if __arm64__
#include <arm64/machine_machdep.h>
#endif

extern kern_return_t kern_invalid(void);

uintptr_t
get_tpidrro(void)
{
	uintptr_t       uthread;
	__asm__ volatile ("mrs %0, TPIDRRO_EL0" : "=r" (uthread));
	return uthread;
}

void
set_tpidrro(uintptr_t uthread)
{
	__asm__ volatile ("msr TPIDRRO_EL0, %0" : : "r" (uthread));
}

kern_return_t
thread_set_cthread_self(vm_address_t self)
{
	return machine_thread_set_tsd_base(current_thread(), self);
}

vm_address_t
thread_get_cthread_self(void)
{
	uintptr_t       self;

	self = get_tpidrro();
	assert( self == current_thread()->machine.cthread_self);
	return self;
}
