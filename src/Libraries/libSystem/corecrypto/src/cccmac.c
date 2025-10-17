/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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

#include <corecrypto/cccmac.h>
#include <corecrypto/ccstubs.h>

int cccmac_init(const struct ccmode_cbc *cbc, cccmac_ctx_t ctx, int key_length, const void *key) {
    CC_STUB_ERR();
}

void cccmac_block_update(const struct ccmode_cbc *cbc, cccmac_ctx_t cmac,
                                       size_t nblocks, const void *data) {
	CC_STUB_VOID();
}


void cccmac_final(const struct ccmode_cbc *cbc, cccmac_ctx_t ctx,
                  size_t nbytes, const void *in, void *out) {
	CC_STUB_VOID();
}

void cccmac(const struct ccmode_cbc *cbc, const void *key,
            size_t data_len, const void *data,
            void *mac) {
	CC_STUB_VOID()
}

void cccmac_one_shot_generate(const struct ccmode_cbc * arg0,
	int key_size,
	const void *key,
	size_t dataLength,
	const uint8_t *data,
	int block_size,
	void *macOut) {
	CC_STUB_VOID();
}

void cccmac_update(cccmac_ctx_t ctx, size_t dataLength, const void *data) {
	CC_STUB_VOID();
}

void cccmac_final_generate(cccmac_ctx_t ctx, size_t dataLength, void *macOut) {
	CC_STUB_VOID();
}

const struct ccmode_cbc* cccmac_cbc(cccmac_ctx_t ctx) {
	CC_STUB(NULL);
}
