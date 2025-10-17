/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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
#ifndef OPENSSL_HEADER_CRYPTO_DILITHIUM_INTERNAL_H
#define OPENSSL_HEADER_CRYPTO_DILITHIUM_INTERNAL_H

#include <openssl/base.h>
#include <openssl/experimental/dilithium.h>

#if defined(__cplusplus)
extern "C" {
#endif


// DILITHIUM_GENERATE_KEY_ENTROPY is the number of bytes of uniformly random
// entropy necessary to generate a key pair.
#define DILITHIUM_GENERATE_KEY_ENTROPY 32

// DILITHIUM_SIGNATURE_RANDOMIZER_BYTES is the number of bytes of uniformly
// random entropy necessary to generate a signature in randomized mode.
#define DILITHIUM_SIGNATURE_RANDOMIZER_BYTES 32

// DILITHIUM_generate_key_external_entropy generates a public/private key pair
// using the given seed, writes the encoded public key to
// |out_encoded_public_key| and sets |out_private_key| to the private key,
// returning 1 on success and 0 on failure. Returns 1 on success and 0 on
// failure.
OPENSSL_EXPORT int DILITHIUM_generate_key_external_entropy(
    uint8_t out_encoded_public_key[DILITHIUM_PUBLIC_KEY_BYTES],
    struct DILITHIUM_private_key *out_private_key,
    const uint8_t entropy[DILITHIUM_GENERATE_KEY_ENTROPY]);

// DILITHIUM_sign_deterministic generates a signature for the message |msg| of
// length |msg_len| using |private_key| following the deterministic algorithm,
// and writes the encoded signature to |out_encoded_signature|. Returns 1 on
// success and 0 on failure.
OPENSSL_EXPORT int DILITHIUM_sign_deterministic(
    uint8_t out_encoded_signature[DILITHIUM_SIGNATURE_BYTES],
    const struct DILITHIUM_private_key *private_key, const uint8_t *msg,
    size_t msg_len);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_DILITHIUM_INTERNAL_H
