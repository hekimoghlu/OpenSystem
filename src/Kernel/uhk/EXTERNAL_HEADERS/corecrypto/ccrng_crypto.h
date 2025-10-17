/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#ifndef _CORECRYPTO_CCRNG_CRYPTO_H_
#define _CORECRYPTO_CCRNG_CRYPTO_H_

#include <corecrypto/ccrng.h>
#include <corecrypto/ccrng_schedule.h>
#include <corecrypto/ccentropy.h>
#include "cc_lock.h"

// This is a framework for a cryptographically-secure RNG. It is
// configurable in many aspects, including:
//
// - DRBG implementation
// - Entropy source
// - Reseed schedule
// - Locks (optional)
// - Request chunking
// - Output caching

#define CCRNG_CRYPTO_SEED_MAX_NBYTES ((size_t)64)

typedef struct ccrng_crypto_ctx {
    CCRNG_STATE_COMMON

    ccentropy_ctx_t *entropy_ctx;
    ccrng_schedule_ctx_t *schedule_ctx;
    cc_lock_ctx_t *lock_ctx;

    const struct ccdrbg_info *drbg_info;
    struct ccdrbg_state *drbg_ctx;

    size_t generate_chunk_nbytes;
    size_t seed_nbytes;

    size_t cache_nbytes;
    uint8_t *cache;
    size_t cache_pos;
} ccrng_crypto_ctx_t;

int
ccrng_crypto_init(ccrng_crypto_ctx_t *ctx,
                  ccentropy_ctx_t *entropy_ctx,
                  ccrng_schedule_ctx_t *schedule_ctx,
                  cc_lock_ctx_t *lock_ctx,
                  const struct ccdrbg_info *drbg_info,
                  struct ccdrbg_state *drbg_ctx,
                  size_t generate_chunk_nbytes,
                  size_t seed_nbytes,
                  size_t cache_nbytes,
                  void *cache);

int
ccrng_crypto_generate(ccrng_crypto_ctx_t *ctx,
                      size_t nbytes,
                      void *rand);

int
ccrng_crypto_reseed(ccrng_crypto_ctx_t *ctx,
                    size_t seed_nbytes,
                    const void *seed,
                    size_t nonce_nbytes,
                    const void *nonce);

#endif /* _CORECRYPTO_CCRNG_CRYPTO_H_ */
