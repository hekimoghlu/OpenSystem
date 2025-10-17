/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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
#include <libkern/crypto/crypto_internal.h>
#include <corecrypto/ccrng.h>
#include <libkern/crypto/rand.h>

int
cc_rand_generate(void *out, size_t outlen)
{
	struct ccrng_state *rng_state = NULL;
	int error = -1;

	if (g_crypto_funcs) {
		rng_state = g_crypto_funcs->ccrng_fn(&error);
		if (rng_state != NULL) {
			error = ccrng_generate(rng_state, outlen, out);
		}
	}

	return error;
}

int
random_buf(void *buf, size_t buflen)
{
	return cc_rand_generate(buf, buflen);
}

void
crypto_random_generate(
	crypto_random_ctx_t ctx,
	void *random,
	size_t random_size)
{
	g_crypto_funcs->random_generate_fn(ctx, random, random_size);
}

void
crypto_random_uniform(
	crypto_random_ctx_t ctx,
	uint64_t bound,
	uint64_t *random)
{
	g_crypto_funcs->random_uniform_fn(ctx, bound, random);
}

size_t
crypto_random_kmem_ctx_size(void)
{
	return g_crypto_funcs->random_kmem_ctx_size_fn();
}

void
crypto_random_kmem_init(
	crypto_random_ctx_t ctx)
{
	g_crypto_funcs->random_kmem_init_fn(ctx);
}
