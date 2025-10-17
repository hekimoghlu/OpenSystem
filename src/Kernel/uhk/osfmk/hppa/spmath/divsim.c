/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
  (c) Copyright 1986 HEWLETT-PACKARD COMPANY
  To anyone who acknowledges that this file is provided "AS IS"
  without any express or implied warranty:
      permission to use, copy, modify, and distribute this file
  for any purpose is hereby granted without fee, provided that
  the above copyright notice and this notice appears in all
  copies, and that the name of Hewlett-Packard Company not be
  used in advertising or publicity pertaining to distribution
  of the software without specific, written prior permission.
  Hewlett-Packard Company makes no representations about the
  suitability of this software for any purpose.
*/
/* @(#)divsim.c: Revision: 1.6.88.1 Date: 93/12/07 15:05:56 */

#include "md.h"

void
divsim(int opnd1, int opnd2, struct mdsfu_register *result)
{
	int sign, op1_sign;

	/* check divisor for zero */
	if (opnd2 == 0) {
		overflow = TRUE;
		return;
	}

	/* get sign of result */
	sign = opnd1 ^ opnd2;

	/* get absolute value of operands */
	if (opnd1 < 0) {
		opnd1 = -opnd1;
		op1_sign = TRUE;
	}
	else op1_sign = FALSE;
	if (opnd2 < 0) opnd2 = -opnd2;

	/* check for opnd2 == -2**31 */
	if (opnd2 < 0) {
		if (opnd1 == opnd2) {
			result_hi = 0;	/* remainder = 0 */
			result_lo = 1;
		}
		else {
			result_hi = opnd1;	/* remainder = opnd1 */
			result_lo = 0;
		}
	}
	else {
		/* do the divide */
		divu(0,opnd1,opnd2,result);

		/*
		 * check for overflow
		 *
		 * at this point, the only way we can get overflow
		 * is with opnd1 = -2**31 and opnd2 = -1
		 */
		if (sign>0 && result_lo<0) {
			overflow = TRUE;
			return;
		}
	}
	overflow = FALSE;

	/* return positive residue */
	if (op1_sign && result_hi) {
		result_hi = opnd2 - result_hi;
		if (++result_lo < 0) {
			overflow = TRUE;
			return;
		}
	}

	/* return appropriately signed result */
	if (sign<0) result_lo = -result_lo;
	return;
}

