/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#include "test_utils.h"

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

static inline uint64_t
xorshift64(uint64_t *state)
{
	uint64_t x = *state;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	*state = x;
	return (x);
}

/*
 * Fill a buffer with reproducible pseudo-random data using a simple xorshift
 * algorithm. Originally, most tests filled buffers with a loop that calls
 * rand() once for each byte. However, this initialization can be extremely
 * slow when running on emulated platforms such as QEMU where 16M calls to
 * rand() take a long time: Before the test_write_format_7zip_large_copy test
 * took ~22 seconds, whereas using a xorshift random number generator (that can
 * be inlined) reduces it to ~17 seconds on QEMU RISC-V.
 */
static void
fill_with_pseudorandom_data_seed(uint64_t seed, void *buffer, size_t size)
{
	uint64_t *aligned_buffer;
	size_t num_values;
	size_t i;
	size_t unaligned_suffix;
	size_t unaligned_prefix = 0;
	/*
	 * To avoid unaligned stores we only fill the aligned part of the buffer
	 * with pseudo-random data and fill the unaligned prefix with 0xab and
	 * the suffix with 0xcd.
	 */
	if ((uintptr_t)buffer % sizeof(uint64_t)) {
		unaligned_prefix =
		    sizeof(uint64_t) - (uintptr_t)buffer % sizeof(uint64_t);
		aligned_buffer =
		    (uint64_t *)((char *)buffer + unaligned_prefix);
		memset(buffer, 0xab, unaligned_prefix);
	} else {
		aligned_buffer = (uint64_t *)buffer;
	}
	assert((uintptr_t)aligned_buffer % sizeof(uint64_t) == 0);
	num_values = (size - unaligned_prefix) / sizeof(uint64_t);
	unaligned_suffix =
	    size - unaligned_prefix - num_values * sizeof(uint64_t);
	for (i = 0; i < num_values; i++) {
		aligned_buffer[i] = xorshift64(&seed);
	}
	if (unaligned_suffix) {
		memset((char *)buffer + size - unaligned_suffix, 0xcd,
		    unaligned_suffix);
	}
}

void
fill_with_pseudorandom_data(void *buffer, size_t size)
{
	uint64_t seed;
	const char* seed_str;
	/*
	 * Check if a seed has been specified in the environment, otherwise fall
	 * back to using rand() as a seed.
	 */
	if ((seed_str = getenv("TEST_RANDOM_SEED")) != NULL) {
		errno = 0;
		seed = strtoull(seed_str, NULL, 10);
		if (errno != 0) {
			fprintf(stderr, "strtoull(%s) failed: %s", seed_str,
			    strerror(errno));
			seed = rand();
		}
	} else {
		seed = rand();
	}
	fill_with_pseudorandom_data_seed(seed, buffer, size);
}
