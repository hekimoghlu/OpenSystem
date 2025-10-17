/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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
 * Copyright (c) 1991 Carnegie Mellon University
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

#ifndef _I386_USER_LDT_H_
#define _I386_USER_LDT_H_

#if defined (__i386__) || defined (__x86_64__)

/*
 * User LDT management.
 *
 * Each task may have its own LDT.
 */

#define LDT_AUTO_ALLOC  0xffffffff

#ifdef KERNEL
#include <i386/seg.h>

struct user_ldt {
	unsigned int start;             /* first descriptor in table */
	unsigned int count;             /* how many descriptors in table */
	struct real_descriptor  ldt[0]; /* descriptor table (variable) */
};
typedef struct user_ldt *       user_ldt_t;

extern user_ldt_t       user_ldt_copy(
	user_ldt_t      uldt);
extern void     user_ldt_free(
	user_ldt_t      uldt);
extern void     user_ldt_set(
	thread_t        thread);
#else /* !KERNEL */
#include <sys/cdefs.h>

union ldt_entry;

__BEGIN_DECLS
int i386_get_ldt(int, union ldt_entry *, int);
int i386_set_ldt(int, const union ldt_entry *, int);
__END_DECLS
#endif /* KERNEL */

#endif /* defined (__i386__) || defined (__x86_64__) */

#endif  /* _I386_USER_LDT_H_ */
