/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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
 * Copyright (c) 1991,1990,1989 Carnegie Mellon University
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

/*
 *	File:	machine/thread.h
 *
 *	This file contains the structure definitions for the thread
 *	state as applied to I386 processors.
 */

#ifndef _I386_THREAD_H_
#define _I386_THREAD_H_

#include <mach/boolean.h>
#include <mach/i386/vm_types.h>
#include <mach/i386/fp_reg.h>
#include <mach/thread_status.h>

#include <kern/simple_lock.h>

#include <i386/fpu.h>
#include <i386/iopb.h>
#include <i386/seg.h>
#include <i386/tss.h>
#include <i386/eflags.h>

#include <i386/cpu_data.h>
#include <i386/proc_reg.h>

#include <machine/pal_routines.h>

/*
 *	machine_thread_kernel_state, x86_kernel_state:
 *
 *	This structure corresponds to the state of kernel registers
 *	as saved in a context-switch.  It lives at the base of the stack.
 */

struct x86_kernel_state {
	uint64_t        k_rbx;  /* kernel context */
	uint64_t        k_rsp;
	uint64_t        k_rbp;
	uint64_t        k_r12;
	uint64_t        k_r13;
	uint64_t        k_r14;
	uint64_t        k_r15;
	uint64_t        k_rip;
};

#ifdef  MACH_KERNEL_PRIVATE
typedef struct x86_kernel_state machine_thread_kernel_state;
#include <kern/thread_kernel_state.h>
#endif

/*
 * Maps state flavor to number of words in the state:
 */
extern unsigned int _MachineStateCount[];

/*
 * The machine-dependent thread state - registers and all platform-dependent
 * state - is saved in the machine thread structure which is embedded in
 * the thread data structure. For historical reasons this is also referred to
 * as the PCB.
 */
struct machine_thread {
	x86_saved_state_t       *iss;
	void                    *ifps;
	void                    *ids;
	decl_simple_lock_data(, lock);           /* protects ifps and ids */
	xstate_t                xstate;

#ifdef  MACH_BSD
	uint64_t                cthread_self;   /* for use of cthread package */
	struct real_descriptor  cthread_desc;
	unsigned long           uldt_selector;  /* user ldt selector to set */
	struct real_descriptor  uldt_desc;      /* actual user setable ldt */
#endif

	struct pal_pcb          pal_pcb;
	uint32_t                specFlags;
	/* N.B.: These "specFlags" are read-modify-written non-atomically within
	 * the copyio routine. So conceivably any exception that modifies the
	 * flags in a persistent manner could be clobbered if it occurs within
	 * a copyio context. For now, the only other flag here is OnProc which
	 * is not modified except at context switch.
	 */
#define         OnProc          0x1
#define         CopyIOActive    0x2 /* Checked to ensure DTrace actions do not re-enter copyio(). */
	uint64_t                thread_gpu_ns;
	uint32_t                last_xcpm_ttd;
	uint8_t                 last_xcpm_index;
	int                     mthr_do_segchk;
#define         MTHR_SEGCHK     1
#define         MTHR_RSBST      2
	int                     insn_state_copyin_failure_errorcode;    /* If insn_state is 0, this may hold the reason */
	x86_instruction_state_t *insn_state;
#if DEVELOPMENT || DEBUG
	/* first byte specifies the offset of the instruction at the time of capture */
	uint8_t                 insn_cacheline[65];     /* XXX: Hard-coded cacheline size */
#endif
	x86_lbrs_t              lbrs;
	bool                    insn_copy_optout;
};
typedef struct machine_thread *pcb_t;

#define THREAD_TO_PCB(Thr)      (&(Thr)->machine)

#define USER_STATE(Thr)         ((Thr)->machine.iss)
#define USER_REGS32(Thr)        (saved_state32(USER_STATE(Thr)))
#define USER_REGS64(Thr)        (saved_state64(USER_STATE(Thr)))

#define user_pc(Thr)            (is_saved_state32(USER_STATE(Thr)) ?    \
	                                USER_REGS32(Thr)->eip :         \
	                                USER_REGS64(Thr)->isf.rip )

extern void *get_user_regs(thread_t);

extern void *act_thread_csave(void);
extern void act_thread_catt(void *ctx);
extern void act_thread_cfree(void *ctx);

#define FIND_PERFCONTROL_STATE(th)      (PERFCONTROL_STATE_NULL)

/*
 *	On the kernel stack is:
 *	stack:	...
 *		struct thread_kernel_state
 *	stack+kernel_stack_size
 */


#define STACK_IKS(stack)        \
	(&(((struct thread_kernel_state *)((stack) + kernel_stack_size)) - 1)->machine)

extern vm_offset_t kernel_stack_size;

/*
 * Return the current stack depth including thread_kernel_state
 *
 * Note: this is only valid to call on a thread's kernel stack,
 * as opposed to the interrupt or special expection stacks, since
 * it's computation is based on cpu_kernel_stack field of the cpu
 * pointer.
 *
 */
static inline vm_offset_t
current_kernel_stack_depth(void)
{
	vm_offset_t     stack_ptr;
	vm_offset_t     stack_depth;

	assert(get_preemption_level() > 0 || !ml_get_interrupts_enabled());

	__asm__ volatile ("mov %%rsp, %0" : "=m" (stack_ptr));

	stack_depth = current_cpu_datap()->cpu_kernel_stack
	    + sizeof(struct thread_kernel_state)
	    - stack_ptr;

	if (stack_depth >= kernel_stack_size) {
		panic("kernel stack overflow; stack base: 0x%16lx, "
		    "stack top: 0x%016lx, stack depth: 0x%016lx, "
		    "depth limit: 0x%016lx", current_cpu_datap()->cpu_kernel_stack,
		    stack_ptr, stack_depth, kernel_stack_size);
	}

	return stack_depth;
}

/*
 * Return address of the function that called current function, given
 *	address of the first parameter of current function.
 */
#define GET_RETURN_PC(addr)     (__builtin_return_address(0))

#endif  /* _I386_THREAD_H_ */
