/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
/* @(#)mpyaccu.c: Revision: 1.6.88.1 Date: 93/12/07 15:06:41 */

#include "md.h"

void
mpyaccu(unsigned int opnd1, unsigned int opnd2, struct mdsfu_register *result)
{
	struct mdsfu_register temp;
	int carry;

	u_xmpy(&opnd1,&opnd2,&temp);

	/* get result of low word add, and check for carry out */
	if ((result_lo += (unsigned)temp.rslt_lo) < (unsigned)temp.rslt_lo)
		carry = 1;
	else
		carry = 0;

	/* get result of high word add, and determine overflow status */
	if ((result_hi += (unsigned)temp.rslt_hi + carry) <
	    (unsigned)temp.rslt_hi)
		overflow = TRUE;
}

