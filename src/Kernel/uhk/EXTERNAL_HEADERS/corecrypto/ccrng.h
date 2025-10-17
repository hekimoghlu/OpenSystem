/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#ifndef _CORECRYPTO_CCRNG_H_
#define _CORECRYPTO_CCRNG_H_

#include <corecrypto/cc.h>

#define CCRNG_STATE_COMMON \
    int (*CC_SPTR(ccrng_state, generate))(struct ccrng_state *rng, size_t outlen, void *out);

/*!
 @type      struct ccrng_state
 @abstract  Default state structure. Do not instantiate. ccrng() returns a reference to this structure
 */
struct ccrng_state {
    CCRNG_STATE_COMMON
};

/*!
 @function ccrng
 @abstract Get a handle to a secure RNG

 @param error A pointer to set in case of error; may be null

 @result A handle to a secure RNG, or null if one cannot be initialized successfully

 @discussion
 This function returns a pointer to the most appropriate RNG for the
 environment. This may be a TRNG if one is available. Otherwise, it is
 a PRNG offering several features:
 - Good performance
 - FIPS Compliant: NIST SP800-90A + FIPS 140-2
 - Seeded from the appropriate entropy source for the platform
 - Provides at least 128-bit security
 - Backtracing resistance
 - Prediction break (after reseed)
 */
struct ccrng_state *ccrng(int *error);

/*!
 @function   ccrng_generate
 @abstract   Generate `outlen` bytes of output, stored in `out`, using ccrng_state `rng`.

 @param rng  `struct ccrng_state` representing the state of the RNG.
 @param outlen  Amount of random bytes to generate.
 @param out  Pointer to memory where random bytes are stored, of size at least `outlen`.

 @result 0 on success and nonzero on failure.
 */
#define ccrng_generate(rng, outlen, out) \
    ((rng)->generate((struct ccrng_state *)(rng), (outlen), (out)))

#if !CC_EXCLAVEKIT
/*!
  @function ccrng_uniform
  @abstract Generate a random value in @p [0, bound).

  @param rng   The state of the RNG.
  @param bound The exclusive upper bound on the output.
  @param rand  A pointer to a single @p uint64_t to store the result.

  @result Returns zero iff the operation is successful.
 */
int ccrng_uniform(struct ccrng_state *rng, uint64_t bound, uint64_t *rand);
#endif

#endif /* _CORECRYPTO_CCRNG_H_ */
