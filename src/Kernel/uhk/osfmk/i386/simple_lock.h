/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
 *	File:	kern/simple_lock_types.h
 *	Author:	Avadis Tevanian, Jr., Michael Wayne Young
 *	Date:	1985
 *
 *	Simple lock data type definitions
 */
#ifdef  KERNEL_PRIVATE

#ifndef _I386_SIMPLE_LOCK_TYPES_H_
#define _I386_SIMPLE_LOCK_TYPES_H_

#include <mach/boolean.h>
#include <kern/lock_types.h>

#include <sys/appleapiopts.h>
#if defined(MACH_KERNEL_PRIVATE) && defined(__APPLE_API_PRIVATE)
#include <mach_ldebug.h>

extern uint64_t LockTimeOutTSC; /* Lock timeout in TSC ticks */
extern uint32_t LockTimeOutUsec;/* Lock timeout in microseconds */
extern uint64_t LockTimeOut;    /* Lock timeout in absolute time */

#if     MACH_LDEBUG
#define USLOCK_DEBUG 1
#else
#define USLOCK_DEBUG 0
#endif  /* USLOCK_DEBUG */

typedef struct uslock_debug {
	void                    *lock_pc;       /* pc where lock operation began    */
	void                    *lock_thread;   /* thread that acquired lock */
	void                    *unlock_thread; /* last thread to release lock */
	void                    *unlock_pc;     /* pc where lock operation ended    */
	unsigned long   duration[2];
	unsigned short  state;
	unsigned char   lock_cpu;
	unsigned char   unlock_cpu;
} uslock_debug;

typedef struct slock {
	hw_lock_data_t  interlock;      /* must be first... see lock.c */
#if     USLOCK_DEBUG
	unsigned short  lock_type;      /* must be second... see lock.c */
#define USLOCK_TAG      0x5353
	uslock_debug    debug;
#endif
} usimple_lock_data_t, *usimple_lock_t;

extern void                     i386_lock_unlock_with_flush(
	hw_lock_t);
#else

typedef struct slock {
	unsigned long   lock_data[10];
} usimple_lock_data_t, *usimple_lock_t;

#endif  /* defined(MACH_KERNEL_PRIVATE) && defined(__APPLE_API_PRIVATE) */

#define USIMPLE_LOCK_NULL       ((usimple_lock_t) 0)

#if !defined(decl_simple_lock_data)
typedef usimple_lock_data_t     *simple_lock_t;
typedef usimple_lock_data_t     simple_lock_data_t;

#define decl_simple_lock_data(class, name) \
	class	simple_lock_data_t	name

#endif  /* !defined(decl_simple_lock_data) */

#endif /* !_I386_SIMPLE_LOCK_TYPES_H_ */

#endif  /* KERNEL_PRIVATE */
