/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#ifndef OPENSSL_HEADER_CRYPTO_SLHDSA_WOTS_H
#define OPENSSL_HEADER_CRYPTO_SLHDSA_WOTS_H

#include "./params.h"

#if defined(__cplusplus)
extern "C" {
#endif


// Implements Algorithm 6: wots_pkGen function, page 18
void slhdsa_wots_pk_gen(uint8_t pk[SLHDSA_SHA2_128S_N],
                        const uint8_t sk_seed[SLHDSA_SHA2_128S_N],
                        const uint8_t pub_seed[SLHDSA_SHA2_128S_N],
                        uint8_t addr[32]);

// Implements Algorithm 7: wots_sign function, page 20
void slhdsa_wots_sign(uint8_t sig[SLHDSA_SHA2_128S_WOTS_BYTES],
                      const uint8_t msg[SLHDSA_SHA2_128S_N],
                      const uint8_t sk_seed[SLHDSA_SHA2_128S_N],
                      const uint8_t pub_seed[SLHDSA_SHA2_128S_N],
                      uint8_t addr[32]);

// Implements Algorithm 8: wots_pkFromSig function, page 21
void slhdsa_wots_pk_from_sig(uint8_t pk[SLHDSA_SHA2_128S_N],
                             const uint8_t sig[SLHDSA_SHA2_128S_WOTS_BYTES],
                             const uint8_t msg[SLHDSA_SHA2_128S_N],
                             const uint8_t pub_seed[SLHDSA_SHA2_128S_N],
                             uint8_t addr[32]);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_SLHDSA_WOTS_H
