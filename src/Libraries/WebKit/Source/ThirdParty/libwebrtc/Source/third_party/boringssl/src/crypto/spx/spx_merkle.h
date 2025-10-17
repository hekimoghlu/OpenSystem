/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
#ifndef OPENSSL_HEADER_CRYPTO_SPX_MERKLE_H
#define OPENSSL_HEADER_CRYPTO_SPX_MERKLE_H

#include <openssl/base.h>

#include <sys/types.h>

#include "./spx_params.h"

#if defined(__cplusplus)
extern "C" {
#endif


// Algorithm 8: Compute the root of a Merkle subtree of WOTS+ public keys.
void spx_treehash(uint8_t out_pk[SPX_N], const uint8_t sk_seed[SPX_N],
                  uint32_t i /*target node index*/,
                  uint32_t z /*target node height*/,
                  const uint8_t pk_seed[SPX_N], uint8_t addr[32]);

// Algorithm 9: Generate an XMSS signature.
void spx_xmss_sign(uint8_t *sig, const uint8_t msg[SPX_N], unsigned int idx,
                   const uint8_t sk_seed[SPX_N], const uint8_t pk_seed[SPX_N],
                   uint8_t addr[32]);

// Algorithm 10: Compute an XMSS public key from an XMSS signature.
void spx_xmss_pk_from_sig(uint8_t *root, const uint8_t *xmss_sig,
                          unsigned int idx, const uint8_t msg[SPX_N],
                          const uint8_t pk_seed[SPX_N], uint8_t addr[32]);

// Algorithm 11: Generate a hypertree signature.
void spx_ht_sign(uint8_t *sig, const uint8_t message[SPX_N], uint64_t idx_tree,
                 uint32_t idx_leaf, const uint8_t sk_seed[SPX_N],
                 const uint8_t pk_seed[SPX_N]);

// Algorithm 12: Verify a hypertree signature.
int spx_ht_verify(const uint8_t sig[SPX_D * SPX_XMSS_BYTES],
                  const uint8_t message[SPX_N], uint64_t idx_tree,
                  uint32_t idx_leaf, const uint8_t pk_root[SPX_N],
                  const uint8_t pk_seed[SPX_N]);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_SPX_MERKLE_H
