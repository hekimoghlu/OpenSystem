/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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

#ifndef _I386_TRAP_H_
#define _I386_TRAP_H_

/*
 * Hardware trap vectors for i386.
 */
#define T_DIVIDE_ERROR          0
#define T_DEBUG                 1
#define T_NMI                   2               /* non-maskable interrupt */
#define T_INT3                  3               /* int 3 instruction */
#define T_OVERFLOW              4               /* overflow test */
#define T_OUT_OF_BOUNDS         5               /* bounds check */
#define T_INVALID_OPCODE        6               /* invalid op code */
#define T_NO_FPU                7               /* no floating point */
#define T_DOUBLE_FAULT          8               /* double fault */
#define T_FPU_FAULT             9
#define T_INVALID_TSS           10
#define T_SEGMENT_NOT_PRESENT   11
#define T_STACK_FAULT           12
#define T_GENERAL_PROTECTION    13
#define T_PAGE_FAULT            14
/*				15 */
#define T_FLOATING_POINT_ERROR  16
#define T_WATCHPOINT            17
#define T_MACHINE_CHECK         18
#define T_SSE_FLOAT_ERROR       19
/*                          20-126 */
#define T_DTRACE_RET            127

/* The SYSENTER and SYSCALL trap numbers are software constructs.
 * These exceptions are dispatched directly to the system call handlers.
 * See also the "software interrupt codes" section of
 * osfmk/mach/i386/syscall_sw.h
 */
#define T_SYSENTER              0x84
#define T_SYSCALL               0x85

#define T_PREEMPT               255

#define TRAP_NAMES "divide error", "debug trap", "NMI", "breakpoint", \
	           "overflow", "bounds check", "invalid opcode", \
	           "no coprocessor", "double fault", "coprocessor overrun", \
	           "invalid TSS", "segment not present", "stack bounds", \
	           "general protection", "page fault", "(reserved)", \
	           "coprocessor error", "watchpoint", "machine check", "SSE floating point"

/*
 * Page-fault trap codes.
 */
#define T_PF_PROT               0x1             /* protection violation */
#define T_PF_WRITE              0x2             /* write access */
#define T_PF_USER               0x4             /* from user state */

#define T_PF_RSVD               0x8             /* reserved bit set to 1 */
#define T_PF_EXECUTE            0x10            /* instruction fetch when NX */

#if !defined(ASSEMBLER)

#define ML_TRAP_REGISTER_1      "rax"
#define ML_TRAP_REGISTER_2      "r10"
#define ML_TRAP_REGISTER_3      "r11"

#define ml_recoverable_trap(code) \
	__asm__ volatile ("ud1l %0(%%eax), %%eax" : : "p"(code))

#define ml_fatal_trap(code)  ({ \
	ml_recoverable_trap(code); \
	__builtin_unreachable(); \
})

#if defined(XNU_KERNEL_PRIVATE)
__attribute__((cold, always_inline))
static inline void
ml_trap(unsigned int code)
{
	__asm__ volatile ("ud1l %0(%%eax), %%eax" : : "p"((void *)((unsigned long long)code)));
}

/* For use by clang option -ftrap-function only */
__attribute__((cold, always_inline))
static inline void
ml_bound_chk_soft_trap(unsigned char code)
{
	/* clang mandates arg to be unsigned char */
	unsigned int code32 = code;
	if (code32 == 0x19) {
		/* if we see a bound check trap, implicitly make it soft */
		code32 += 0xFF00; /* code defined in kern/telemetry.h */
	}

	/* let other codes fall through */
	ml_trap(code32);
}
#endif /* XNU_KERNEL_PRIVATE */
#endif /* !ASSEMBLER */

#endif  /* _I386_TRAP_H_ */
