/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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
/* @(#)hppa.h: Revision: 2.7.88.1 Date: 93/12/07 15:06:26 */

/* amount is assumed to be a constant between 0 and 32 (non-inclusive) */
#define Shiftdouble(left,right,amount,dest)			\
    /* int left, right, amount, dest; */			\
    dest = ((left) << (32-(amount))) | ((unsigned int)(right) >> (amount))

/* amount must be less than 32 */
#define Variableshiftdouble(left,right,amount,dest)		\
    /* unsigned int left, right;  int amount, dest; */		\
    if (amount == 0) dest = right;				\
    else dest = ((((unsigned) left)&0x7fffffff) << (32-(amount))) |	\
	((unsigned) right >> (amount))

/* amount must be between 0 and 32 (non-inclusive) */
#define Variable_shift_double(left,right,amount,dest)		\
    /* unsigned int left, right;  int amount, dest; */		\
    dest = (left << (32-(amount))) | ((unsigned) right >> (amount))
