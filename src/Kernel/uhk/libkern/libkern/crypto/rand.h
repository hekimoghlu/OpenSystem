/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#ifndef _RAND_H
#define _RAND_H

#include <libkern/crypto/crypto.h>

__BEGIN_DECLS

// A handle to a random generator suitable for use with
// crypto_random_generate.
typedef void *crypto_random_ctx_t;

// The maximum size (in bytes) of a random generator.
#define CRYPTO_RANDOM_MAX_CTX_SIZE ((size_t)256)

typedef void (*crypto_random_generate_fn_t)(
	crypto_random_ctx_t ctx,
	void *random,
	size_t random_size);

typedef void (*crypto_random_uniform_fn_t)(
	crypto_random_ctx_t ctx,
	uint64_t bound,
	uint64_t *random);

typedef size_t (*crypto_random_kmem_ctx_size_fn_t)(void);

typedef void (*crypto_random_kmem_init_fn_t)(
	crypto_random_ctx_t ctx);

#if XNU_KERNEL_PRIVATE

int cc_rand_generate(void *out, size_t outlen);

// Generate random data with the supplied handle to a random
// generator. The behavior of this function (e.g. the quality of the
// randomness, whether it might acquire a lock, the cryptographic
// primitives used) depends on the semantics of the generator.
void crypto_random_generate(
	crypto_random_ctx_t ctx,
	void *random,
	size_t random_size);

// Generate a random value in the range [0, bound), i.e. including
// zero and excluding the bound. The generated value is stored in the
// random pointer which should point to a single value. As above, the
// behavior of this function depends in part on the semantics of the
// generator.
void crypto_random_uniform(
	crypto_random_ctx_t ctx,
	uint64_t bound,
	uint64_t *random);

// The following two functions are for use in the kmem subsystem
// only. They are NOT guaranteed to provide cryptographic randomness
// and should not be used elsewhere.

// Return the size needed for a random generator to be used by
// kmem. (See the discussion below for the semantics of this
// generator.)
//
// The returned value may vary by platform, but it is guaranteed to be
// no larger than CRYPTO_RANDOM_MAX_CTX_SIZE.
size_t crypto_random_kmem_ctx_size(void);

// Initialize the handle with a random generator for use by kmem. This
// function should only be called by kmem.
//
// The handle should point to memory at least as large as
// crypto_random_kmem_ctx_size() indicates.
//
// This generator is NOT guaranteed to provide cryptographic
// randomness.
//
// The initialized generator is guaranteed not to acquire a
// lock. (Note, however, that this initialization function MAY acquire
// a lock.)
//
// The initialized generator is guaranteed not to touch FP registers
// on Intel.
void crypto_random_kmem_init(
	crypto_random_ctx_t ctx);

#endif  /* XNU_KERNEL_PRIVATE */

int random_buf(void *buf, size_t buflen);

__END_DECLS

#endif
