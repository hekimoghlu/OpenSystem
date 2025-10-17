/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#ifndef OPENSSL_HEADER_CRYPTO_FIPSMODULE_SLHDSA_FORS_H
#define OPENSSL_HEADER_CRYPTO_FIPSMODULE_SLHDSA_FORS_H

#include "./params.h"

#if defined(__cplusplus)
extern "C" {
#endif


// Implements Algorithm 14: fors_skGen function (page 29)
void slhdsa_fors_sk_gen(uint8_t fors_sk[BCM_SLHDSA_SHA2_128S_N], uint32_t idx,
                        const uint8_t sk_seed[BCM_SLHDSA_SHA2_128S_N],
                        const uint8_t pk_seed[BCM_SLHDSA_SHA2_128S_N],
                        uint8_t addr[32]);

// Implements Algorithm 15: fors_node function (page 30)
void slhdsa_fors_treehash(uint8_t root_node[BCM_SLHDSA_SHA2_128S_N],
                          const uint8_t sk_seed[BCM_SLHDSA_SHA2_128S_N],
                          uint32_t i /*target node index*/,
                          uint32_t z /*target node height*/,
                          const uint8_t pk_seed[BCM_SLHDSA_SHA2_128S_N],
                          uint8_t addr[32]);

// Implements Algorithm 16: fors_sign function (page 31)
void slhdsa_fors_sign(uint8_t fors_sig[SLHDSA_SHA2_128S_FORS_BYTES],
                      const uint8_t message[SLHDSA_SHA2_128S_FORS_MSG_BYTES],
                      const uint8_t sk_seed[BCM_SLHDSA_SHA2_128S_N],
                      const uint8_t pk_seed[BCM_SLHDSA_SHA2_128S_N],
                      uint8_t addr[32]);

// Implements Algorithm 17: fors_pkFromSig function (page 32)
void slhdsa_fors_pk_from_sig(
    uint8_t fors_pk[BCM_SLHDSA_SHA2_128S_N],
    const uint8_t fors_sig[SLHDSA_SHA2_128S_FORS_BYTES],
    const uint8_t message[SLHDSA_SHA2_128S_FORS_MSG_BYTES],
    const uint8_t pk_seed[BCM_SLHDSA_SHA2_128S_N], uint8_t addr[32]);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_FIPSMODULE_SLHDSA_FORS_H
