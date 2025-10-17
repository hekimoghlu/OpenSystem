/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

#include <corecrypto/ccmode_factory.h>

int ccmode_cbc_init(const struct ccmode_cbc* cbc, cccbc_ctx* _ctx, size_t rawkey_len, const void* rawkey) {
	int status = CCERR_OK;
	struct _ccmode_cbc_key* ctx = (struct _ccmode_cbc_key*)_ctx;
	ctx->ecb = cbc->custom;

	ccecb_ctx* ecb_ctx = (ccecb_ctx*)((char*)ctx->u + ccn_sizeof_size(ctx->ecb->block_size));
	if ((status = ctx->ecb->init(ctx->ecb, ecb_ctx, rawkey_len, rawkey)) != CCERR_OK)
		goto out;

	// `_ccmode_cbc_key` includes space for a single block in the context, which i *think* is supposed to be scratch space.
	cc_zero(ccn_sizeof_size(ctx->ecb->block_size), ctx->u);

out:
	return status;
};

int ccmode_cbc_encrypt(const cccbc_ctx* _ctx, cccbc_iv* _iv, size_t nblocks, const void* in, void* out) {
	int status = CCERR_OK;
	struct _ccmode_cbc_key* ctx = (struct _ccmode_cbc_key*)_ctx;
	ccecb_ctx* ecb_ctx = (ccecb_ctx*)((char*)ctx->u + ccn_sizeof_size(ctx->ecb->block_size));
	const uint8_t* input = in;
	uint8_t* output = out;
	uint8_t* scratch_space = (uint8_t*)ctx->u;
	uint8_t* iv = (uint8_t*)_iv;

	for (size_t i = 0; i < nblocks; ++i) {
		cc_xor(ctx->ecb->block_size, scratch_space, input, iv);

		if ((status = ctx->ecb->ecb(ecb_ctx, 1, scratch_space, output)) != CCERR_OK)
			goto out;

		cc_copy(ctx->ecb->block_size, iv, output);

		input += ctx->ecb->block_size;
		output += ctx->ecb->block_size;
	}

out:
	return status;
};

int ccmode_cbc_decrypt(const cccbc_ctx* _ctx, cccbc_iv* _iv, size_t nblocks, const void* in, void* out) {
	int status = CCERR_OK;
	struct _ccmode_cbc_key* ctx = (struct _ccmode_cbc_key*)_ctx;
	ccecb_ctx* ecb_ctx = (ccecb_ctx*)((char*)ctx->u + ccn_sizeof_size(ctx->ecb->block_size));
	const uint8_t* input = in;
	uint8_t* output = out;
	uint8_t* scratch_space = (uint8_t*)ctx->u;
	uint8_t* iv = (uint8_t*)_iv;

	for (size_t i = 0; i < nblocks; ++i) {
		if ((status = ctx->ecb->ecb(ecb_ctx, 1, input, scratch_space)) != CCERR_OK)
			goto out;

		cc_xor(ctx->ecb->block_size, output, scratch_space, iv);

		cc_copy(ctx->ecb->block_size, iv, input);

		input += ctx->ecb->block_size;
		output += ctx->ecb->block_size;
	}

out:
	return status;
};
