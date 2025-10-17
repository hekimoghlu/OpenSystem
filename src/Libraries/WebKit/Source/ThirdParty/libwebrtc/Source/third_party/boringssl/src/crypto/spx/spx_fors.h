/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
#ifndef OPENSSL_HEADER_CRYPTO_SPX_FORS_H
#define OPENSSL_HEADER_CRYPTO_SPX_FORS_H

#include <openssl/base.h>

#include "./spx_params.h"

#if defined(__cplusplus)
extern "C" {
#endif


// Algorithm 13: Generate a FORS private key value.
void spx_fors_sk_gen(uint8_t *fors_sk, uint32_t idx,
                     const uint8_t sk_seed[SPX_N], const uint8_t pk_seed[SPX_N],
                     uint8_t addr[32]);

// Algorithm 14: Compute the root of a Merkle subtree of FORS public values.
void spx_fors_treehash(uint8_t root_node[SPX_N], const uint8_t sk_seed[SPX_N],
                       uint32_t i /*target node index*/,
                       uint32_t z /*target node height*/,
                       const uint8_t pk_seed[SPX_N], uint8_t addr[32]);

// Algorithm 15: Generate a FORS signature.
void spx_fors_sign(uint8_t *fors_sig, const uint8_t message[SPX_FORS_MSG_BYTES],
                   const uint8_t sk_seed[SPX_N], const uint8_t pk_seed[SPX_N],
                   uint8_t addr[32]);

// Algorithm 16: Compute a FORS public key from a FORS signature.
void spx_fors_pk_from_sig(uint8_t *fors_pk,
                          const uint8_t fors_sig[SPX_FORS_BYTES],
                          const uint8_t message[SPX_FORS_MSG_BYTES],
                          const uint8_t pk_seed[SPX_N], uint8_t addr[32]);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_SPX_FORS_H
