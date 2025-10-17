/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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
 * @OSF_COPYRIGHT@
 */
/*
 * Mach Operating System
 * Copyright (c) 1991,1990 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
/*
 */

#ifndef _I386_TRAP_INTERNAL_H_
#define _I386_TRAP_INTERNAL_H_

#include <i386/trap.h>
#include <i386/thread.h>

#define DEFAULT_PANIC_ON_TRAP_MASK ((1U << T_INVALID_OPCODE) |  \
	(1U << T_GENERAL_PROTECTION) |                          \
	(1U << T_PAGE_FAULT) |                                  \
	(1U << T_SEGMENT_NOT_PRESENT) |                         \
	(1U << T_STACK_FAULT))


extern void             i386_exception(
	int                     exc,
	mach_exception_code_t   code,
	mach_exception_subcode_t subcode);

extern void             sync_iss_to_iks(x86_saved_state_t *regs);

extern void             sync_iss_to_iks_unconditionally(
	x86_saved_state_t       *regs);

extern void             kernel_trap(x86_saved_state_t *regs, uintptr_t *lo_spp);

extern void             user_trap(x86_saved_state_t *regs);

extern void             interrupt(x86_saved_state_t *regs);

extern void             panic_double_fault64(x86_saved_state_t *regs) __abortlike;
extern void             panic_machine_check64(x86_saved_state_t *regs) __abortlike;

typedef kern_return_t (*perfCallback)(
	int                     trapno,
	void                    *regs,
	uintptr_t               *lo_spp,
	int);

extern void             panic_i386_backtrace(void *, int, const char *, boolean_t, x86_saved_state_t *);
extern void     print_one_backtrace(pmap_t pmap, vm_offset_t topfp, const char *cur_marker, boolean_t is_64_bit);
extern void     print_thread_num_that_crashed(task_t task);
extern void     print_tasks_user_threads(task_t task);
extern void     print_threads_registers(thread_t thread);
extern void     print_uuid_info(task_t task);
extern void     print_launchd_info(void);

#if MACH_KDP
extern boolean_t        kdp_i386_trap(
	unsigned int,
	x86_saved_state64_t *,
	kern_return_t,
	vm_offset_t);
#endif /* MACH_KDP */

#endif /* _I386_TRAP_INTERNAL_H_ */
