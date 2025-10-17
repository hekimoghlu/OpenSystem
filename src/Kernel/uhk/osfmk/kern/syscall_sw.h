/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
 * Copyright (c) 1991,1990,1989,1988,1987 Carnegie Mellon University
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

#ifndef _KERN_SYSCALL_SW_H_
#define _KERN_SYSCALL_SW_H_

#include <mach_assert.h>
#include <mach/kern_return.h>

/*
 *	mach_trap_stack indicates the trap may discard
 *	its kernel stack.  Some architectures may need
 *	to save more state in the pcb for these traps.
 */

/* Keep this in sync with DTrace's version in bsd/dev/dtrace/systrace.c */
#if CONFIG_REQUIRES_U32_MUNGING
typedef void mach_munge_t(void *);
#elif __arm__ && (__BIGGEST_ALIGNMENT__ > 4)
typedef int mach_munge_t(const void *, void *);
#endif

typedef struct {
	unsigned char           mach_trap_arg_count; /* Number of trap arguments (Arch independant) */
	unsigned char           mach_trap_u32_words; /* number of 32-bit words to copyin for U32 */
	unsigned char           mach_trap_returns_port;
	unsigned char           __mach_trap_padding;
	kern_return_t         (*mach_trap_function)(void *);
#if CONFIG_REQUIRES_U32_MUNGING || (__arm__ && (__BIGGEST_ALIGNMENT__ > 4))
	mach_munge_t           *mach_trap_arg_munge32; /* system call argument munger routine for 32-bit */
#endif
#if MACH_ASSERT
	const char             *mach_trap_name;
#endif /* MACH_ASSERT */
} mach_trap_t;

#define MACH_TRAP_TABLE_COUNT   128

extern const mach_trap_t        mach_trap_table[MACH_TRAP_TABLE_COUNT];
extern const int                mach_trap_count;
extern const char * const       mach_syscall_name_table[MACH_TRAP_TABLE_COUNT];

#if CONFIG_REQUIRES_U32_MUNGING || (__arm__ && (__BIGGEST_ALIGNMENT__ > 4))
#define MACH_TRAP_MUNGE(arg)    .mach_trap_arg_munge32 = arg,
#else
#define MACH_TRAP_MUNGE(arg)
#endif

#if MACH_ASSERT
#define MACH_TRAP_NAME(arg)     .mach_trap_name = #arg,
#else
#define MACH_TRAP_NAME(arg)
#endif

#define MACH_TRAP(name, arg_count, u32_arg_words, munge32, ...)  { \
	.mach_trap_arg_count = arg_count, \
	.mach_trap_u32_words = u32_arg_words, \
	.mach_trap_function = (kern_return_t (*)(void *)) (name), \
	MACH_TRAP_MUNGE(munge32) \
	MACH_TRAP_NAME(name) \
	__VA_ARGS__ \
}

#endif  /* _KERN_SYSCALL_SW_H_ */
