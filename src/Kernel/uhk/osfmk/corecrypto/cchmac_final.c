/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#include <corecrypto/ccdigest_priv.h>
#include <corecrypto/cchmac.h>
#include <corecrypto/ccn.h>
#include <corecrypto/cc_priv.h>

void
cchmac_final(const struct ccdigest_info *di, cchmac_ctx_t hc,
    unsigned char *mac)
{
	CC_ENSURE_DIT_ENABLED


	// Finalize the inner state of the data being HMAC'd, i.e., H((key \oplus ipad) || m)
	ccdigest_final(di, cchmac_digest_ctx(di, hc), cchmac_data(di, hc));

	// Set the HMAC output size based on the digest algorithm
	cchmac_num(di, hc) = (unsigned int)di->output_size; /* typecast: output size will always fit in an unsigned int */
	cchmac_nbits(di, hc) = di->block_size * 8;

	// Copy the pre-computed compress(key \oplus opad) back to digest state,
	// and then run through the digest once more to finish the HMAC
	ccdigest_copy_state(di, cchmac_istate32(di, hc), cchmac_ostate32(di, hc));
	ccdigest_final(di, cchmac_digest_ctx(di, hc), mac);
}

