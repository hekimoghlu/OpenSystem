/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include <openssl/base.h>

#include <string.h>

#define OPENSSL_UNSTABLE_EXPERIMENTAL_SPX
#include <openssl/experimental/spx.h>
#include <openssl/rand.h>

#include "./spx_address.h"
#include "./spx_fors.h"
#include "./spx_merkle.h"
#include "./spx_params.h"
#include "./spx_util.h"
#include "./spx_thash.h"

void SPX_generate_key(uint8_t out_public_key[SPX_PUBLIC_KEY_BYTES],
                      uint8_t out_secret_key[SPX_SECRET_KEY_BYTES]) {
  uint8_t seed[3 * SPX_N];
  RAND_bytes(seed, 3 * SPX_N);
  SPX_generate_key_from_seed(out_public_key, out_secret_key, seed);
}

void SPX_generate_key_from_seed(uint8_t out_public_key[SPX_PUBLIC_KEY_BYTES],
                                uint8_t out_secret_key[SPX_SECRET_KEY_BYTES],
                                const uint8_t seed[3 * SPX_N]) {
  // Initialize SK.seed || SK.prf || PK.seed from seed.
  memcpy(out_secret_key, seed, 3 * SPX_N);

  // Initialize PK.seed from seed.
  memcpy(out_public_key, seed + 2 * SPX_N, SPX_N);

  uint8_t addr[32] = {0};
  spx_set_layer_addr(addr, SPX_D - 1);

  // Set PK.root
  spx_treehash(out_public_key + SPX_N, out_secret_key, 0, SPX_TREE_HEIGHT,
               out_public_key, addr);
  memcpy(out_secret_key + 3 * SPX_N, out_public_key + SPX_N, SPX_N);
}

void SPX_sign(uint8_t out_signature[SPX_SIGNATURE_BYTES],
              const uint8_t secret_key[SPX_SECRET_KEY_BYTES],
              const uint8_t *msg, size_t msg_len, int randomized) {
  uint8_t addr[32] = {0};
  const uint8_t *sk_seed = secret_key;
  const uint8_t *sk_prf = secret_key + SPX_N;
  const uint8_t *pk_seed = secret_key + 2 * SPX_N;
  const uint8_t *pk_root = secret_key + 3 * SPX_N;

  uint8_t opt_rand[SPX_N] = {0};

  if (randomized) {
    RAND_bytes(opt_rand, SPX_N);
  } else {
    memcpy(opt_rand, pk_seed, SPX_N);
  }

  // Derive randomizer r and copy it to signature.
  uint8_t r[SPX_N];
  spx_thash_prfmsg(r, sk_prf, opt_rand, msg, msg_len);
  memcpy(out_signature, r, SPX_N);

  uint8_t digest[SPX_DIGEST_SIZE];
  spx_thash_hmsg(digest, r, pk_seed, pk_root, msg, msg_len);

  uint8_t fors_digest[SPX_FORS_MSG_BYTES];
  memcpy(fors_digest, digest, SPX_FORS_MSG_BYTES);

  uint8_t *tmp_idx_tree = digest + SPX_FORS_MSG_BYTES;
  uint8_t *tmp_idx_leaf = tmp_idx_tree + SPX_TREE_BYTES;

  uint64_t idx_tree = spx_to_uint64(tmp_idx_tree, SPX_TREE_BYTES);
  idx_tree &= (~(uint64_t)0) >> (64 - SPX_TREE_BITS);

  uint32_t idx_leaf = (uint32_t)spx_to_uint64(tmp_idx_leaf, SPX_LEAF_BYTES);
  idx_leaf &= (~(uint32_t)0) >> (32 - SPX_LEAF_BITS);

  spx_set_tree_addr(addr, idx_tree);
  spx_set_type(addr, SPX_ADDR_TYPE_FORSTREE);
  spx_set_keypair_addr(addr, idx_leaf);

  spx_fors_sign(out_signature + SPX_N, fors_digest, sk_seed, pk_seed, addr);

  uint8_t pk_fors[SPX_N];
  spx_fors_pk_from_sig(pk_fors, out_signature + SPX_N, fors_digest, pk_seed,
                       addr);

  spx_ht_sign(out_signature + SPX_N + SPX_FORS_BYTES, pk_fors, idx_tree,
              idx_leaf, sk_seed, pk_seed);
}

int SPX_verify(const uint8_t signature[SPX_SIGNATURE_BYTES],
               const uint8_t public_key[SPX_SECRET_KEY_BYTES],
               const uint8_t *msg, size_t msg_len) {
  uint8_t addr[32] = {0};
  const uint8_t *pk_seed = public_key;
  const uint8_t *pk_root = public_key + SPX_N;

  const uint8_t *r = signature;
  const uint8_t *sig_fors = signature + SPX_N;
  const uint8_t *sig_ht = sig_fors + SPX_FORS_BYTES;

  uint8_t digest[SPX_DIGEST_SIZE];
  spx_thash_hmsg(digest, r, pk_seed, pk_root, msg, msg_len);

  uint8_t fors_digest[SPX_FORS_MSG_BYTES];
  memcpy(fors_digest, digest, SPX_FORS_MSG_BYTES);

  uint8_t *tmp_idx_tree = digest + SPX_FORS_MSG_BYTES;
  uint8_t *tmp_idx_leaf = tmp_idx_tree + SPX_TREE_BYTES;

  uint64_t idx_tree = spx_to_uint64(tmp_idx_tree, SPX_TREE_BYTES);
  idx_tree &= (~(uint64_t)0) >> (64 - SPX_TREE_BITS);

  uint32_t idx_leaf = (uint32_t)spx_to_uint64(tmp_idx_leaf, SPX_LEAF_BYTES);
  idx_leaf &= (~(uint32_t)0) >> (32 - SPX_LEAF_BITS);

  spx_set_tree_addr(addr, idx_tree);
  spx_set_type(addr, SPX_ADDR_TYPE_FORSTREE);
  spx_set_keypair_addr(addr, idx_leaf);

  uint8_t pk_fors[SPX_N];
  spx_fors_pk_from_sig(pk_fors, sig_fors, fors_digest, pk_seed, addr);

  return spx_ht_verify(sig_ht, pk_fors, idx_tree, idx_leaf, pk_root, pk_seed);
}
