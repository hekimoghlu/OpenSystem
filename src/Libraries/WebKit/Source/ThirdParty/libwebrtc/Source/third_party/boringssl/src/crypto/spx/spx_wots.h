/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#ifndef OPENSSL_HEADER_CRYPTO_SPX_WOTS_H
#define OPENSSL_HEADER_CRYPTO_SPX_WOTS_H

#include <openssl/base.h>

#include "./spx_params.h"

#if defined(__cplusplus)
extern "C" {
#endif


// Algorithm 5: Generate a WOTS+ public key.
void spx_wots_pk_gen(uint8_t *pk, const uint8_t sk_seed[SPX_N],
                     const uint8_t pub_seed[SPX_N], uint8_t addr[32]);

// Algorithm 6: Generate a WOTS+ signature on an n-byte message.
void spx_wots_sign(uint8_t *sig, const uint8_t msg[SPX_N],
                   const uint8_t sk_seed[SPX_N], const uint8_t pub_seed[SPX_N],
                   uint8_t addr[32]);

// Algorithm 7: Compute a WOTS+ public key from a message and its signature.
void spx_wots_pk_from_sig(uint8_t *pk, const uint8_t *sig, const uint8_t *msg,
                          const uint8_t pub_seed[SPX_N], uint8_t addr[32]);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_SPX_WOTS_H
