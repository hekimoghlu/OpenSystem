/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
#ifndef _RAND48_H_
#define _RAND48_H_

#include <math.h>
#include <stdlib.h>

#define	RAND48_SEED_0	(0x330e)
#define	RAND48_SEED_1	(0xabcd)
#define	RAND48_SEED_2	(0x1234)
#define	RAND48_MULT_0	(0xe66d)
#define	RAND48_MULT_1	(0xdeec)
#define	RAND48_MULT_2	(0x0005)
#define	RAND48_ADD	(0x000b)

typedef unsigned long long uint48;

extern uint48 _rand48_seed;
extern uint48 _rand48_mult;
extern uint48 _rand48_add;

#define TOUINT48(x,y,z) \
	((uint48)(x) + (((uint48)(y)) << 16) + (((uint48)(z)) << 32))

#define RAND48_SEED	TOUINT48(RAND48_SEED_0, RAND48_SEED_1, RAND48_SEED_2)
#define RAND48_MULT	TOUINT48(RAND48_MULT_0, RAND48_MULT_1, RAND48_MULT_2)

#define LOADRAND48(l,x) \
	(l) = TOUINT48((x)[0], (x)[1], (x)[2])

#define STORERAND48(l,x) \
	(x)[0] = (unsigned short)(l); \
	(x)[1] = (unsigned short)((l) >> 16); \
	(x)[2] = (unsigned short)((l) >> 32)

#define _DORAND48(l) \
	(l) = (l) * _rand48_mult + _rand48_add

#define DORAND48(l,x) \
	LOADRAND48(l, x); \
	_DORAND48(l); \
	STORERAND48(l, x)

#include "fpmath.h"

/*
 * Optimization for speed: avoid int-to-double conversion.  Assume doubles
 * are IEEE-754 and insert the bits directly.  To normalize, the (1 << 52)
 * is the hidden bit, which the first set bit is shifted to.
 */
#define ERAND48_BEGIN \
	union { \
		union IEEEd2bits ieee; \
		unsigned long long l; \
	} u; \
	int s

#define ERAND48_END(x) \
	u.l = ((x) & 0xffffffffffffULL); \
	if (u.l == 0) \
		return 0.0; \
	u.l <<= 5; \
	for(s = 0; !(u.l & (1LL << 52)); s++, u.l <<= 1) {} \
	u.ieee.bits.exp = 1022 - s; \
	return u.ieee.d

#endif /* _RAND48_H_ */
