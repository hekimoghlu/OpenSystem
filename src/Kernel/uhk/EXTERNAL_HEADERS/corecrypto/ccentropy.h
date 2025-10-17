/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#ifndef _CORECRYPTO_CCENTROPY_H_
#define _CORECRYPTO_CCENTROPY_H_

#include <corecrypto/cc.h>
#include <corecrypto/ccdigest.h>

// An interface to provide high-entropy seeds to RNGs.

typedef struct ccentropy_ctx ccentropy_ctx_t;

typedef int (*ccentropy_get_seed_fn_t)(ccentropy_ctx_t *ctx,
                                       size_t seed_nbytes,
                                       void *seed);

typedef int (*ccentropy_add_entropy_fn_t)(ccentropy_ctx_t *ctx,
                                          uint32_t entropy_nsamples,
                                          size_t entropy_nbytes,
                                          const void *entropy);

// A descriptor for an entropy implementation.
typedef struct ccentropy_info {
    // This is a required function. Implementations should populate
    // the seed with a full-entropy output. If they are temporarily
    // unable due to insufficient entropy, they should return
    // CCERR_OUT_OF_ENTROPY. If they are permanently unable they
    // should return some other error (or abort).
    ccentropy_get_seed_fn_t get_seed;

    // This is an optional function. The caller will provide a set of
    // (potentially low-quality) entropy samples, and the
    // implementation should mix these into its internal
    // state. Implementations are free to omit this function if it
    // does not make sense (e.g. see ccentropy_rng below).
    ccentropy_add_entropy_fn_t add_entropy;
} ccentropy_info_t;

// Common state for entropy implementations.
struct ccentropy_ctx {
    // A pointer to the descriptor.
    const ccentropy_info_t *info;
};

/*!
  @function ccentropy_get_seed
  @abstract Get a high-entropy seed.

  @param ctx The entropy context.
  @param seed_nbytes The size of the seed requested.
  @param seed A buffer to receive the seed.

  @return CCERR_OK on success; CCERR_OUT_OF_ENTROPY if entropy is
  temporarily unavailable; some implementation-defined error (or
  abort) otherwise.
*/
int ccentropy_get_seed(ccentropy_ctx_t *ctx,
                       size_t seed_nbytes,
                       void *seed);

/*!
  @function ccentropy_add_entropy
  @abstract Add fresh entropy samples to the context.

  @param ctx The entropy context.
  @param entropy_nsamples The count of samples included in this batch.
  @param entropy_nbytes The size of the entropy payload in bytes.
  @param entropy A buffer containing the fresh entropy samples.

  @return CCERR_OK on success; CCERR_NOT_SUPPORTED if this operation
  is not supported for the implementation; some implementation-defined
  error (or abort) otherwise.

  @discussion This operation is optional and will not be supported by
  all implementations.
*/
int ccentropy_add_entropy(ccentropy_ctx_t *ctx,
                          uint32_t entropy_nsamples,
                          size_t entropy_nbytes,
                          const void *entropy);

// A simple wrapper around a ccrng instance. This implementation does
// not support the add_entropy interface.
typedef struct ccentropy_rng_ctx {
    ccentropy_ctx_t entropy_ctx;
    struct ccrng_state *rng_ctx;
    size_t seed_max_nbytes;
} ccentropy_rng_ctx_t;

/*!
  @function ccentropy_rng_init
  @abstract Wrap a ccrng instance in the ccentropy interface.

  @param ctx The entropy context.
  @param rng_ctx The RNG to wrap.
  @param seed_max_nbytes The maximum seed size that this RNG can provide.

  @return CCERR_OK on success.

  @discussion seed_max_nbytes should correspond to the security level
  of the underlying RNG.
*/
int ccentropy_rng_init(ccentropy_rng_ctx_t *ctx,
                       struct ccrng_state *rng_ctx,
                       size_t seed_max_nbytes);

// An entropy conditioner based on digest functions. We assume a fixed
// per-sample entropy estimate measured in millibits
// (i.e. mbits). This estimate should be determined via offline
// analysis.
typedef struct ccentropy_digest_ctx {
    ccentropy_ctx_t entropy_ctx;
    const struct ccdigest_info *digest_info;
    ccdigest_ctx_decl(MAX_DIGEST_STATE_SIZE,
                      MAX_DIGEST_BLOCK_SIZE,
                      digest_ctx);
    uint32_t entropy_mbits_per_sample;
    uint32_t entropy_mbits;
} ccentropy_digest_ctx_t;

#define CCENTROPY_MBITS_PER_BYTE ((uint32_t)(8000))

/*!
  @function ccentropy_digest_init
  @abstract Initialize a digest-based entropy conditioner.

  @param ctx The entropy context.
  @param digest_info A descriptor for the digest.
  @param entropy_mbits_per_sample An estimate of per-sample entropy measured in millibits.

  @return CCERR_OK on success.

  @discussion The estimated entropy per sample should be determined
  via offline analysis.
*/
int ccentropy_digest_init(struct ccentropy_digest_ctx *ctx,
                          const struct ccdigest_info *digest_info,
                          uint32_t entropy_mbits_per_sample);

#endif /* _CORECRYPTO_CCENTROPY_H_ */
