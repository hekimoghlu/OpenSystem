/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#include <corecrypto/ccdigest_priv.h>
#include <corecrypto/cc_priv.h>
#include "ccdigest_internal.h"

/* This can be used for SHA1, SHA256 and SHA224 */
void
ccdigest_final_64be(const struct ccdigest_info *di, ccdigest_ctx_t ctx, unsigned char *digest)
{
	// Sanity check to recover from ctx corruptions.
	if (ccdigest_num(di, ctx) >= di->block_size) {
		ccdigest_num(di, ctx) = 0;
	}

	// Clone the state.
	ccdigest_di_decl(di, tmp);
	cc_memcpy(tmp, ctx, ccdigest_di_size(di));

	ccdigest_nbits(di, tmp) += ccdigest_num(di, tmp) * 8;
	ccdigest_data(di, tmp)[ccdigest_num(di, tmp)++] = 0x80;

	/* If we don't have at least 8 bytes (for the length) left we need to add
	 *  a second block. */
	if (ccdigest_num(di, tmp) > 64 - 8) {
		while (ccdigest_num(di, tmp) < 64) {
			ccdigest_data(di, tmp)[ccdigest_num(di, tmp)++] = 0;
		}
		di->compress(ccdigest_state(di, tmp), 1, ccdigest_data(di, tmp));
		ccdigest_num(di, tmp) = 0;
	}

	/* pad upto block_size minus 8 with 0s */
	while (ccdigest_num(di, tmp) < 64 - 8) {
		ccdigest_data(di, tmp)[ccdigest_num(di, tmp)++] = 0;
	}

	cc_store64_be(ccdigest_nbits(di, tmp), ccdigest_data(di, tmp) + 64 - 8);
	di->compress(ccdigest_state(di, tmp), 1, ccdigest_data(di, tmp));

	/* copy output */
	for (unsigned int i = 0; i < di->output_size / 4; i++) {
		cc_store32_be(ccdigest_state_u32(di, tmp)[i], digest + (4 * i));
	}

	ccdigest_di_clear(di, tmp);
}

