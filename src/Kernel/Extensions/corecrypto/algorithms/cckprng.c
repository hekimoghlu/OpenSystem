/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
#include <stddef.h>
#include <corecrypto/cckprng.h>
#include <corecrypto/ccrng.h>
#include "yarrow/yarrow.h"
#if !KERNEL
#include <stdint.h>
#endif

void cckprng_init(struct cckprng_ctx *ctx, unsigned max_ngens, size_t entropybuf_nbytes, const void *entropybuf,
				  const uint32_t *entropybuf_nsamples, size_t seed_nbytes, const void *seed, size_t nonce_nbytes,
				  const void *nonce) {
	prngInitialize(&ctx->prng);
	ctx->bytes_generated = ctx->bytes_since_entropy = 0;

	cckprng_reseed(ctx, seed_nbytes, seed);
}

void cckprng_initgen(struct cckprng_ctx *ctx, unsigned gen_idx) {
	// Nothing is needed here.
}

void cckprng_reseed(struct cckprng_ctx *ctx, size_t nbytes, const void *seed) {
	prngInput(ctx->prng, (BYTE *)seed, (UINT)nbytes, 0, 0);
	prngAllowReseed(ctx->prng, 5000);
	ctx->bytes_since_entropy = 0;
}

void cckprng_refresh(struct cckprng_ctx *ctx) {
	// Nothing is needed here, either.
}

void cckprng_generate(struct cckprng_ctx *ctx, unsigned gen_idx, size_t nbytes, void *out) {
	BYTE *buffer = (BYTE *)out;
	while (nbytes > UINT32_MAX) {
		prngOutput(ctx->prng, buffer, UINT32_MAX);
		ctx->bytes_generated += UINT32_MAX;
		ctx->bytes_since_entropy += UINT32_MAX;

		buffer += UINT32_MAX;
		nbytes -= UINT32_MAX;
		nbytes = CC_MAX(nbytes, 0);
	}

	if (nbytes != 0) {
		prngOutput(ctx->prng, buffer, (UINT)nbytes);
		ctx->bytes_generated += nbytes;
		ctx->bytes_since_entropy += nbytes;
	}
}

// MARK: -

struct ccrng_impl {
	CCRNG_STATE_COMMON;
	PrngRef prng;
};

static int ccrng_generate_impl(struct ccrng_state *rng, size_t outlen, void *out) {
	struct ccrng_impl *impl = (struct ccrng_impl *)rng;
	prngOutput(impl->prng, out, (UINT)outlen);
	return 0;
}

struct ccrng_state *ccrng(int *error) {
	static struct ccrng_impl ccrng_impl = { ccrng_generate_impl, NULL };
	static bool initialized = false;

	if (!initialized) {
		prngInitialize(&ccrng_impl.prng);
		initialized = true;
	}

	return (struct ccrng_state *)&ccrng_impl;
}
