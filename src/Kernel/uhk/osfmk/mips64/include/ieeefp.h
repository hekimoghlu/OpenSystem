/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
 * Written by J.T. Conklin, Apr 11, 1995
 * Public domain.
 */

#ifndef _MIPS64_IEEEFP_H_
#define _MIPS64_IEEEFP_H_

typedef int fp_except;
#define FP_X_IMP	0x01	/* imprecise (loss of precision) */
#define FP_X_UFL	0x02	/* underflow exception */
#define FP_X_OFL	0x04	/* overflow exception */
#define FP_X_DZ		0x08	/* divide-by-zero exception */
#define FP_X_INV	0x10	/* invalid operation exception */

typedef enum {
    FP_RN=0,			/* round to nearest representable number */
    FP_RZ=1,			/* round to zero (truncate) */
    FP_RP=2,			/* round toward positive infinity */
    FP_RM=3			/* round toward negative infinity */
} fp_rnd;

#ifdef _KERNEL

/*
 * Defines for the floating-point completion/emulation code.
 */

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/proc.h>
#include <machine/fpu.h>

#define	float_raise(bits) \
	do { curproc->p_md.md_regs->fsr |= (bits) << FPCSR_C_SHIFT; } while (0)
#define	float_set_inexact()	float_raise(FP_X_IMP)
#define	float_set_invalid()	float_raise(FP_X_INV)

#define	float_get_round(csr)	(csr & FPCSR_RM_MASK)
#define	fpgetround()		float_get_round(curproc->p_md.md_regs->fsr)

#endif

#endif /* !_MIPS64_IEEEFP_H_ */
