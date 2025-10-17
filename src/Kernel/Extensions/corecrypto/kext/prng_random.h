/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#ifndef _PRNG_RANDOM_H_
#define _PRNG_RANDOM_H_

__BEGIN_DECLS

#include <corecrypto/cckprng.h>

#ifdef XNU_KERNEL_PRIVATE

#define ENTROPY_BUFFER_BYTE_SIZE 32

#define ENTROPY_BUFFER_SIZE (ENTROPY_BUFFER_BYTE_SIZE / sizeof(uint32_t))

// This mask can be applied to EntropyData.sample_count to get an
// index suitable for storing the next sample in
// EntropyData.buffer. Note that ENTROPY_BUFFER_SIZE must be a power
// of two for the following mask calculation to be valid.
#define ENTROPY_BUFFER_INDEX_MASK (ENTROPY_BUFFER_SIZE - 1)

typedef struct entropy_data {
	/*
	 * TODO: Should sample_count be volatile?  Are we exposed to any races that
	 * we care about if it is not?
	 */

	// At 32 bits, this counter can overflow. Since we're primarily
	// interested in the delta from one read to the next, we don't
	// worry about this too much.
	uint32_t sample_count;
	uint32_t buffer[ENTROPY_BUFFER_SIZE];
} entropy_data_t;

extern entropy_data_t EntropyData;

/* Trace codes for DBG_SEC_KERNEL: */
#define ENTROPY_READ(n) SECURITYDBG_CODE(DBG_SEC_KERNEL, n) /* n: 0 .. 3 */

void random_cpu_init(int cpu);


#endif /* XNU_KERNEL_PRIVATE */

void register_and_init_prng(struct cckprng_ctx *ctx, const struct cckprng_funcs *funcs);

__END_DECLS

#endif /* _PRNG_RANDOM_H_ */
