/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
/*	$NetBSD: pcb.h,v 1.5 1996/11/13 22:21:00 cgd Exp $	*/

/*
 * Copyright (c) 1994, 1995, 1996 Carnegie-Mellon University.
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

#ifndef _MACHINE_PCB_H_
#define _MACHINE_PCB_H_

#include <machine/frame.h>
#include <machine/reg.h>

#include <machine/alpha_cpu.h>

/*
 * PCB: process control block
 *
 * In this case, the hardware structure that is the defining element
 * for a process, and the additional state that must be saved by software
 * on a context switch.  Fields marked [HW] are mandated by hardware; fields
 * marked [SW] are for the software.
 *
 * It's said in the VMS PALcode section of the AARM that the pcb address
 * passed to the swpctx PALcode call has to be a physical address.  Not
 * knowing this (and trying a virtual) address proved this correct.
 * So we cache the physical address of the pcb in the md_proc struct.
 */
struct pcb {
	struct alpha_pcb pcb_hw;		/* PALcode defined */
	unsigned long	pcb_context[9];		/* s[0-6], ra, ps	[SW] */
	struct fpreg	pcb_fp;			/* FP registers		[SW] */
	unsigned long	pcb_onfault;		/* for copy faults	[SW] */
	struct cpu_info *volatile pcb_fpcpu;	/* CPU with our FP state[SW] */
};

#ifdef _KERNEL
void savectx(struct pcb *);
#endif

#endif /* _MACHINE_PCB_H_ */
