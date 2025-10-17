/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#include "cc_internal.h"
#include <corecrypto/ccdigest.h>
#include <corecrypto/cc_priv.h>

void
ccdigest_update(const struct ccdigest_info *di, ccdigest_ctx_t ctx, size_t len, const void *data)
{
	CC_ENSURE_DIT_ENABLED

	const char * data_ptr = data;
	size_t nblocks, nbytes;

	// Sanity check to recover from ctx corruptions.
	if (ccdigest_num(di, ctx) >= di->block_size) {
		ccdigest_num(di, ctx) = 0;
	}

	while (len > 0) {
		if (ccdigest_num(di, ctx) == 0 && len > di->block_size) {
			if (di->block_size == 1 << 6) { // md5 & sha1 & sha256
				nblocks = len >> 6;
				nbytes = nblocks << 6;
			} else if (di->block_size == 1 << 7) { // sha384 & sha512
				nblocks = len >> 7;
				nbytes = nblocks << 7;
			} else {
				nblocks = len / di->block_size;
				nbytes = nblocks * di->block_size;
			}

			di->compress(ccdigest_state(di, ctx), nblocks, data_ptr);
			len -= nbytes;
			data_ptr += nbytes;
			ccdigest_nbits(di, ctx) += (uint64_t) (nbytes) * 8;
		} else {
			size_t n = CC_MIN(di->block_size - ccdigest_num(di, ctx), len);
			cc_memcpy(ccdigest_data(di, ctx) + ccdigest_num(di, ctx), data_ptr, n);
			/* typecast: less than block size, will always fit into an int */
			ccdigest_num(di, ctx) += (unsigned int)n;
			len -= n;
			data_ptr += n;
			if (ccdigest_num(di, ctx) == di->block_size) {
				di->compress(ccdigest_state(di, ctx), 1, ccdigest_data(di, ctx));
				ccdigest_nbits(di, ctx) += ccdigest_num(di, ctx) * 8;
				ccdigest_num(di, ctx) = 0;
			}
		}
	}
}

