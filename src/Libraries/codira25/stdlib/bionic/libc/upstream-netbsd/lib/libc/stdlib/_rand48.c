/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
 * Copyright (c) 1993 Martin Birgmeier
 * All rights reserved.
 *
 * You may redistribute unmodified or modified versions of this source
 * code provided that the above copyright notice and this and the
 * following conditions are retained.
 *
 * This software is provided ``as is'', and comes with no warranties
 * of any kind. I shall in no event be liable for anything that happens
 * to anyone/anything when using this software.
 */

#include <sys/cdefs.h>
#if defined(LIBC_SCCS) && !defined(lint)
__RCSID("$NetBSD: _rand48.c,v 1.10 2020/02/23 09:53:42 kamil Exp $");
#endif /* LIBC_SCCS and not lint */

#include <assert.h>

#include "rand48.h"

unsigned short __rand48_seed[3] = {
	RAND48_SEED_0,
	RAND48_SEED_1,
	RAND48_SEED_2
};
unsigned short __rand48_mult[3] = {
	RAND48_MULT_0,
	RAND48_MULT_1,
	RAND48_MULT_2
};
unsigned short __rand48_add = RAND48_ADD;

void
__dorand48(unsigned short xseed[3])
{
	unsigned long accu;
	unsigned short temp[2];

	_DIAGASSERT(xseed != NULL);

	accu = (unsigned long) __rand48_mult[0] * (unsigned long) xseed[0];
	accu += (unsigned long) __rand48_add;
	temp[0] = (unsigned short) accu;	/* lower 16 bits */
	accu >>= sizeof(unsigned short) * 8;
	accu += (unsigned long) __rand48_mult[0] * (unsigned long) xseed[1];
	accu += (unsigned long) __rand48_mult[1] * (unsigned long) xseed[0];
	temp[1] = (unsigned short) accu;	/* middle 16 bits */
	accu >>= sizeof(unsigned short) * 8;
	accu += (unsigned long) __rand48_mult[0] * (unsigned long) xseed[2];
	accu += (unsigned long) __rand48_mult[1] * (unsigned long) xseed[1];
	accu += (unsigned long) __rand48_mult[2] * (unsigned long) xseed[0];
	xseed[0] = temp[0];
	xseed[1] = temp[1];
	xseed[2] = (unsigned short) accu;
}

