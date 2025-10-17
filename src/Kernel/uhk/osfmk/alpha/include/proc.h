/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
/*	$NetBSD: proc.h,v 1.2 1995/03/24 15:01:36 cgd Exp $	*/

/*
 * Copyright (c) 1994, 1995 Carnegie-Mellon University.
 * All rights reserved.
 *
 * Author: Chris G. Demetriou
 * 
 * Permission to use, copy, modify and distribute this software and
 * its documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 * 
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS" 
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND 
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

#include <machine/cpu.h>
/*
 * Machine-dependent part of the proc struct for the Alpha.
 */

struct mdbpt {
	vaddr_t	addr;
	u_int32_t contents;
};

struct mdproc {
	u_int md_flags;
	volatile u_int md_astpending;	/* AST pending for this process */
	struct trapframe *md_tf;	/* trap/syscall registers */
	struct pcb *md_pcbpaddr;	/* phys addr of the pcb */
	struct mdbpt md_sstep[2];	/* two breakpoints for sstep */
};

/*
 * md_flags usage
 * --------------
 * MDP_FPUSED
 *      A largely unused bit indicating the presence of FPU history.
 *      Cleared on exec. Set but not used by the fpu context switcher
 *      itself.
 *
 * MDP_FP_C
 *      The architected FP Control word. It should forever begin at bit 1,
 *      as the bits are AARM specified and this way it doesn't need to be
 *      shifted.
 *
 *      Until C99 there was never an IEEE 754 API, making most of the
 *      standard useless.  Because of overlapping AARM, OSF/1, NetBSD, and
 *      C99 API's, the use of the MDP_FP_C bits is defined variously in
 *      ieeefp.h and fpu.h.
 */
#define	MDP_FPUSED	0x00000001		/* Process used the FPU */
#ifndef NO_IEEE
#define	MDP_FP_C	0x007ffffe	/* Extended FP_C Quadword bits */
#endif
#define MDP_STEP1	0x00800000	/* Single step normal */
#define MDP_STEP2	0x01800000	/* Single step branch */
