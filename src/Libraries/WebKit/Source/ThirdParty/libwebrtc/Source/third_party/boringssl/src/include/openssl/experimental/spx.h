/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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
#ifndef OPENSSL_HEADER_SPX_H
#define OPENSSL_HEADER_SPX_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


#if defined(OPENSSL_UNSTABLE_EXPERIMENTAL_SPX)
// This header implements experimental, draft versions of not-yet-standardized
// primitives. When the standard is complete, these functions will be removed
// and replaced with the final, incompatible standard version. They are
// available now for short-lived experiments, but must not be deployed anywhere
// durable, such as a long-lived key store. To use these functions define
// OPENSSL_UNSTABLE_EXPERIMENTAL_SPX

// SPX_N is the number of bytes in the hash output
#define SPX_N 16

// SPX_PUBLIC_KEY_BYTES is the nNumber of bytes in the public key of
// SPHINCS+-SHA2-128s
#define SPX_PUBLIC_KEY_BYTES 32

// SPX_SECRET_KEY_BYTES is the number of bytes in the private key of
// SPHINCS+-SHA2-128s
#define SPX_SECRET_KEY_BYTES 64

// SPX_SIGNATURE_BYTES is the number of bytes in a signature of
// SPHINCS+-SHA2-128s
#define SPX_SIGNATURE_BYTES 7856

// SPX_generate_key generates a SPHINCS+-SHA2-128s key pair and writes the
// result to |out_public_key| and |out_secret_key|.
// Private key: SK.seed || SK.prf || PK.seed || PK.root
// Public key: PK.seed || PK.root
OPENSSL_EXPORT void SPX_generate_key(
    uint8_t out_public_key[SPX_PUBLIC_KEY_BYTES],
    uint8_t out_secret_key[SPX_SECRET_KEY_BYTES]);

// SPX_generate_key_from_seed generates a SPHINCS+-SHA2-128s key pair from a
// 48-byte seed and writes the result to |out_public_key| and |out_secret_key|.
// Secret key: SK.seed || SK.prf || PK.seed || PK.root
// Public key: PK.seed || PK.root
OPENSSL_EXPORT void SPX_generate_key_from_seed(
    uint8_t out_public_key[SPX_PUBLIC_KEY_BYTES],
    uint8_t out_secret_key[SPX_SECRET_KEY_BYTES],
    const uint8_t seed[3 * SPX_N]);

// SPX_sign generates a SPHINCS+-SHA2-128s signature over |msg| or length
// |msg_len| using |secret_key| and writes the output to |out_signature|.
//
// if |randomized| is 0, deterministic signing is performed, otherwise,
// non-deterministic signing is performed.
OPENSSL_EXPORT void SPX_sign(
    uint8_t out_snignature[SPX_SIGNATURE_BYTES],
    const uint8_t secret_key[SPX_SECRET_KEY_BYTES], const uint8_t *msg,
    size_t msg_len, int randomized);

// SPX_verify verifies a SPHINCS+-SHA2-128s signature in |signature| over |msg|
// or length |msg_len| using |public_key|. 1 is returned if the signature
// matches, 0 otherwise.
OPENSSL_EXPORT int SPX_verify(
    const uint8_t signature[SPX_SIGNATURE_BYTES],
    const uint8_t public_key[SPX_SECRET_KEY_BYTES], const uint8_t *msg,
    size_t msg_len);

#endif //OPENSSL_UNSTABLE_EXPERIMENTAL_SPX


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_SPX_H
