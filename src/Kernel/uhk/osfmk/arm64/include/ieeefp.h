/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
 * Based on ieeefp.h written by J.T. Conklin, Apr 28, 1995
 * Public domain.
 */

#ifndef _MACHINE_IEEEFP_H_
#define _MACHINE_IEEEFP_H_

typedef int fp_except;
#define	FP_X_INV	0x01	/* invalid operation exception */
#define	FP_X_DZ		0x02	/* divide-by-zero exception */
#define	FP_X_OFL	0x04	/* overflow exception */
#define	FP_X_UFL	0x08	/* underflow exception */
#define	FP_X_IMP	0x10	/* imprecise (loss of precision) */

typedef enum {
    FP_RN=0,			/* round to nearest representable number */
    FP_RP=1,			/* round toward positive infinity */
    FP_RM=2,			/* round toward negative infinity */
    FP_RZ=3			/* round to zero (truncate) */
} fp_rnd;

#endif /* _MACHINE_IEEEFP_H_ */
