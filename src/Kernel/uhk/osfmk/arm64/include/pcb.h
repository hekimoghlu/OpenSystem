/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
 * Copyright (c) 2016 Dale Rahn <drahn@dalerahn.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */
#ifndef	_MACHINE_PCB_H_
#define	_MACHINE_PCB_H_

#include <machine/frame.h>
#include <machine/reg.h>

struct trapframe;

/*
 * Warning certain fields must be within 256 bytes of the beginning
 * of this structure.
 */
struct pcb {
	u_int		pcb_flags;
#define	PCB_FPU		0x00000001	/* Process had FPU initialized */
#define	PCB_SINGLESTEP	0x00000002	/* Single step process */
#define	PCB_SVE		0x00000004	/* Process had SVE initialized */
	struct		trapframe *pcb_tf;

	register_t	pcb_sp;		/* stack pointer of switchframe */

	caddr_t		pcb_onfault;	/* On fault handler */
	struct fpreg	pcb_fpstate;	/* Floating Point state */
	__uint16_t	pcb_sve_p[16];	/* SVE predicate registers */
	__uint16_t	pcb_sve_ffr;	/* SVE first fault register */

	void		*pcb_tcb;
};
#endif	/* _MACHINE_PCB_H_ */
