/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
/* public domain */
#ifndef	_MACHINE_CPU_H_
#define	_MACHINE_CPU_H_

#include <m88k/asm_macro.h>
#include <m88k/cpu.h>

#ifdef _KERNEL

#define	ci_curspl	ci_cpudep4
#define	ci_swireg	ci_cpudep5
#define	ci_intr_mask	ci_cpudep6
#define	ci_clock_ack	ci_cpudep7

void luna88k_ext_int(struct trapframe *eframe);
#define	md_interrupt_func	luna88k_ext_int

static inline u_long
intr_disable(void)
{
	u_long psr;

	psr = get_psr();
	set_psr(psr | PSR_IND);
	return psr;
}

static inline void
intr_restore(u_long psr)
{
	set_psr(psr);
}

#endif	/* _KERNEL */

#endif	/* _MACHINE_CPU_H_ */
