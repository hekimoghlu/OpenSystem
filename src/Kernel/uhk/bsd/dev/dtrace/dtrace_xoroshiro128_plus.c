/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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
#include <sys/types.h>

#include "dtrace_xoroshiro128_plus.h"

static __inline uint64_t
rotl(const uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

/*
 * This is the jump function for the generator. It is equivalent to 2^64 calls
 * to next(); it can be used to generate 2^64 non-overlapping subsequences for
 * parallel computations.
 */
void
dtrace_xoroshiro128_plus_jump(uint64_t * const state,
    uint64_t * const jump_state)
{
	static const uint64_t JUMP[] = { 0xbeac0467eba5facb,
		                         0xd86b048b86aa9922 };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	size_t i = 0;
	int b = 0;
	for (i = 0; i < sizeof JUMP / sizeof *JUMP; i++) {
		for (b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b) {
				s0 ^= state[0];
				s1 ^= state[1];
			}
			dtrace_xoroshiro128_plus_next(state);
		}
	}
	jump_state[0] = s0;
	jump_state[1] = s1;
}

/*
 * xoroshiro128+ - XOR/rotate/shift/rotate
 * xorshift.di.unimi.it
 */
uint64_t
dtrace_xoroshiro128_plus_next(uint64_t * const state)
{
	const uint64_t s0 = state[0];
	uint64_t s1 = state[1];
	uint64_t result;
	result = s0 + s1;

	s1 ^= s0;
	state[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
	state[1] = rotl(s1, 36);

	return result;
}
