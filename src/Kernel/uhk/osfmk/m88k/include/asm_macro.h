/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
 * Mach Operating System
 * Copyright (c) 1993-1991 Carnegie Mellon University
 * Copyright (c) 1991 OMRON Corporation
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON AND OMRON ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON AND OMRON DISCLAIM ANY LIABILITY OF ANY KIND
 * FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 */

#ifndef _M88K_ASM_MACRO_H_
#define _M88K_ASM_MACRO_H_

/*
 * Various compiler macros used for speed and efficiency.
 * Anyone can include.
 */

/*
 * Flush the data pipeline.
 */
#define	flush_pipeline() \
	__asm__ volatile ("tb1 0, %%r0, 0" ::: "memory")

/*
 * Set the PSR.
 */
static __inline__ void
set_psr(u_int psr)
{
	__asm__ volatile ("stcr %0, %%cr1" :: "r" (psr));
	flush_pipeline();
}

/*
 * Get the PSR.
 */
static __inline__ u_int
get_psr(void)
{
	u_int psr;
	__asm__ volatile ("ldcr %0, %%cr1" : "=r" (psr));
	return (psr);
}

/*
 * Provide access from C code to the assembly instruction ff1
 */
static __inline__ u_int
ff1(u_int val)
{
	__asm__ volatile ("ff1 %0, %0" : "=r" (val) : "0" (val));
	return (val);
}

static __inline__ u_int
get_cpu_pid(void)
{
	u_int pid;
	__asm__ volatile ("ldcr %0, %%cr0" : "=r" (pid));
	return (pid);
}

#endif /* _M88K_ASM_MACRO_H_ */
