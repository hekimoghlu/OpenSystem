/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#include "BigNum128.h"


U128 UInt64mult(const uint64_t A, const uint64_t B)
{
	U128	result;
	
#ifndef __OPEN_SOURCE__							// <rdar://13613944>
	// Karatsuba multiplication
	// Suppose we want to multiply two 2 numbers A * B, where A = a1 << 32 + a0, B = b1 << 32 + b0:
	// 1. compute a1 * b1, call the result X
	// 2. compute a0 * b0, call the result Y
	// 3. compute Z, this number is equal to a1 * b0 + a0 * b1.
	// 4. compute X << 64 + Z << 32 + Y
#endif

	uint64_t a1, a0, b1, b0;
	a1 = A >> 32;
	a0 = A - (a1 << 32);
	b1 = B >> 32;
	b0 = B - (b1 << 32);
	
	uint64_t X, Y, Z;
	X = a1 * b1;
	Y = a0 * b0;
	Z = a1 * b0 + a0 * b1;
	
	uint64_t z1, z0;
	z1 = Z >> 32;
	z0 = Z - (z1 << 32);
	
	return U128(X, 0) + U128(z1, uint64_t(z0) << 32) + U128(0, Y);
}
