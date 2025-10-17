/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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

#ifndef _I386_EFLAGS_H_
#define _I386_EFLAGS_H_

#if defined (__i386__) || defined (__x86_64__)

/*
 *	i386 flags register
 */

#ifndef EFL_CF
#define EFL_CF          0x00000001              /* carry */
#define EFL_PF          0x00000004              /* parity of low 8 bits */
#define EFL_AF          0x00000010              /* carry out of bit 3 */
#define EFL_ZF          0x00000040              /* zero */
#define EFL_SF          0x00000080              /* sign */
#define EFL_TF          0x00000100              /* trace trap */
#define EFL_IF          0x00000200              /* interrupt enable */
#define EFL_DF          0x00000400              /* direction */
#define EFL_OF          0x00000800              /* overflow */
#define EFL_IOPL        0x00003000              /* IO privilege level: */
#define EFL_IOPL_KERNEL 0x00000000                      /* kernel */
#define EFL_IOPL_USER   0x00003000                      /* user */
#define EFL_NT          0x00004000              /* nested task */
#define EFL_RF          0x00010000              /* resume without tracing */
#define EFL_VM          0x00020000              /* virtual 8086 mode */
#define EFL_AC          0x00040000              /* alignment check */
#define EFL_VIF         0x00080000              /* virtual interrupt flag */
#define EFL_VIP         0x00100000              /* virtual interrupt pending */
#define EFL_ID          0x00200000              /* cpuID instruction */
#endif

#define EFL_CLR         0xfff88028
#define EFL_SET         0x00000002

#define EFL_USER_SET    (EFL_IF)
#define EFL_USER_CLEAR  (EFL_IOPL|EFL_NT|EFL_RF)

#endif /* defined (__i386__) || defined (__x86_64__) */

#endif  /* _I386_EFLAGS_H_ */
