/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#ifndef OPENSSL_HEADER_CRYPTO_SLHDSA_INTERNAL_H
#define OPENSSL_HEADER_CRYPTO_SLHDSA_INTERNAL_H

#include <openssl/slhdsa.h>

#include "params.h"

#if defined(__cplusplus)
extern "C" {
#endif


// SLHDSA_SHA2_128S_generate_key_from_seed generates an SLH-DSA-SHA2-128s key
// pair from a 48-byte seed and writes the result to |out_public_key| and
// |out_secret_key|.
OPENSSL_EXPORT void SLHDSA_SHA2_128S_generate_key_from_seed(
    uint8_t out_public_key[SLHDSA_SHA2_128S_PUBLIC_KEY_BYTES],
    uint8_t out_secret_key[SLHDSA_SHA2_128S_PRIVATE_KEY_BYTES],
    const uint8_t seed[3 * SLHDSA_SHA2_128S_N]);

// SLHDSA_SHA2_128S_sign_internal acts like |SLHDSA_SHA2_128S_sign| but
// accepts an explicit entropy input, which can be PK.seed (bytes 32..48 of
// the private key) to generate deterministic signatures. It also takes the
// input message in three parts so that the "internal" version of the signing
// function, from section 9.2, can be implemented. The |header| argument may be
// NULL to omit it.
OPENSSL_EXPORT void SLHDSA_SHA2_128S_sign_internal(
    uint8_t out_signature[SLHDSA_SHA2_128S_SIGNATURE_BYTES],
    const uint8_t secret_key[SLHDSA_SHA2_128S_PRIVATE_KEY_BYTES],
    const uint8_t header[SLHDSA_M_PRIME_HEADER_LEN], const uint8_t *context,
    size_t context_len, const uint8_t *msg, size_t msg_len,
    const uint8_t entropy[SLHDSA_SHA2_128S_N]);

// SLHDSA_SHA2_128S_verify_internal acts like |SLHDSA_SHA2_128S_verify| but
// takes the input message in three parts so that the "internal" version of the
// verification function, from section 9.3, can be implemented. The |header|
// argument may be NULL to omit it.
OPENSSL_EXPORT int SLHDSA_SHA2_128S_verify_internal(
    const uint8_t *signature, size_t signature_len,
    const uint8_t public_key[SLHDSA_SHA2_128S_PUBLIC_KEY_BYTES],
    const uint8_t header[SLHDSA_M_PRIME_HEADER_LEN], const uint8_t *context,
    size_t context_len, const uint8_t *msg, size_t msg_len);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_SLHDSA_INTERNAL_H
