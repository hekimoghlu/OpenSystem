/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
#ifndef OPENSSL_HEADER_CRYPTO_SPX_ADDRESS_H
#define OPENSSL_HEADER_CRYPTO_SPX_ADDRESS_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


#define SPX_ADDR_TYPE_WOTS 0
#define SPX_ADDR_TYPE_WOTSPK 1
#define SPX_ADDR_TYPE_HASHTREE 2
#define SPX_ADDR_TYPE_FORSTREE 3
#define SPX_ADDR_TYPE_FORSPK 4
#define SPX_ADDR_TYPE_WOTSPRF 5
#define SPX_ADDR_TYPE_FORSPRF 6

void spx_set_chain_addr(uint8_t addr[32], uint32_t chain);
void spx_set_hash_addr(uint8_t addr[32], uint32_t hash);
void spx_set_keypair_addr(uint8_t addr[32], uint32_t keypair);
void spx_set_layer_addr(uint8_t addr[32], uint32_t layer);
void spx_set_tree_addr(uint8_t addr[32], uint64_t tree);
void spx_set_type(uint8_t addr[32], uint32_t type);
void spx_set_tree_height(uint8_t addr[32], uint32_t tree_height);
void spx_set_tree_index(uint8_t addr[32], uint32_t tree_index);
void spx_copy_keypair_addr(uint8_t out[32], const uint8_t in[32]);

uint32_t spx_get_tree_index(uint8_t addr[32]);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_SPX_ADDRESS_H
