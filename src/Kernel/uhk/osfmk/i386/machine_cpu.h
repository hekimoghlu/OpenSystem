/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
#ifndef _I386_MACHINE_CPU_H_
#define _I386_MACHINE_CPU_H_

#include <mach/mach_types.h>
#include <mach/boolean.h>
#include <kern/kern_types.h>
#include <pexpert/pexpert.h>
#include <sys/cdefs.h>

__BEGIN_DECLS
void    cpu_machine_init(
	void);

void    handle_pending_TLB_flushes(
	void);

int cpu_signal_handler(x86_saved_state_t *regs);

kern_return_t cpu_register(
	int *slot_nump);
__END_DECLS

static inline void
cpu_halt(void)
{
	asm volatile ( "wbinvd; cli; hlt");
}

static inline void
cpu_pause(void)
{
	__builtin_ia32_pause();
}

#endif /* _I386_MACHINE_CPU_H_ */
