/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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

#include <stdint.h>

#include <corecrypto/cc.h>

#define CC_ERR_DEVICE                   -100
#define CC_ERR_INTERUPTS                -101
#define CC_ERR_CRYPTO_CONFIG            -102
#define CC_ERR_PERMS                    -103
#define CC_ERR_PARAMETER                -104
#define CC_ERR_MEMORY                   -105
#define CC_ERR_FILEDESC                 -106
#define CC_ERR_OUT_OF_ENTROPY           -107
#define CC_ERR_INTERNAL                 -108
#define CC_ERR_ATFORK                   -109
#define CC_ERR_OVERFLOW                 -110

#define CCRNG_STATE_COMMON                                                          \
    int (*generate)(struct ccrng_state *rng, size_t outlen, void *out);

/* Get a pointer to a ccrng has never been simpler! Just call this */
struct ccrng_state *ccrng(int *error);

/* default state structure - do not instantiate, instead use the specific one you need */
struct ccrng_state {
    CCRNG_STATE_COMMON
};

#define ccrng_generate(ctx, outlen, out) ((ctx)->generate((ctx), (outlen), (out)))

/* Generate a random value in [0, bound) */
int ccrng_uniform(struct ccrng_state *rng, uint64_t bound, uint64_t *rand);

#endif /* _CORECRYPTO_CCRNG_H_ */
