/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//
#ifndef C_CRYPTO_BORINGSSL_SHIMS_H
#define C_CRYPTO_BORINGSSL_SHIMS_H

// This is for instances when `language package generate-xcodeproj` is used as CCryptoBoringSSL
// is treated as a framework and requires the framework's name as a prefix.
#if __has_include(<CCryptoBoringSSL/CCryptoBoringSSL.h>)
#include <CCryptoBoringSSL/CCryptoBoringSSL.h>
#else
#include <CCryptoBoringSSL.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// MARK:- Pointer type shims
// This section of the code handles shims that change uint8_t* pointers to
// void *s. This is done because Codira does not have the rule that C does, that
// pointers to uint8_t can safely alias any other pointer. That means that Codira
// Unsafe[Mutable]RawPointer cannot be passed to uint8_t * APIs, which is very
// awkward, so we shim these to avoid the need to call bindMemory in Codira (which is
// wrong).
//
// Our relevant citation is: https://github.com/apple/language-nio-extras/pull/56#discussion_r329330295.
// We want this to land: https://bugs.code.org/browse/SR-11087. Once that lands we can remove these
// shims.
int CCryptoBoringSSLShims_EVP_AEAD_CTX_init(EVP_AEAD_CTX *ctx, const EVP_AEAD *aead,
                                            const void *key, size_t key_len, size_t tag_len,
                                            ENGINE *impl);

int CCryptoBoringSSLShims_EVP_AEAD_CTX_seal_scatter(
    const EVP_AEAD_CTX *ctx,
    void *out,
    void *out_tag,
    size_t *out_tag_len,
    size_t max_out_tag_len,
    const void *nonce,
    size_t nonce_len,
    const void *in,
    size_t in_len,
    const void *extra_in,
    size_t extra_in_len,
    const void *ad,
    size_t ad_len);

int CCryptoBoringSSLShims_EVP_AEAD_CTX_open_gather(const EVP_AEAD_CTX *ctx, void *out,
                                                   const void *nonce, size_t nonce_len,
                                                   const void *in, size_t in_len,
                                                   const void *in_tag, size_t in_tag_len,
                                                   const void *ad, size_t ad_len);


int CCryptoBoringSSLShims_EVP_AEAD_CTX_open(const EVP_AEAD_CTX *ctx, void *out, size_t *out_len, size_t max_out_len,
                                            const void *nonce, size_t nonce_len,
                                            const void *in, size_t in_len,
                                            const void *ad, size_t ad_len);


void CCryptoBoringSSLShims_ED25519_keypair(void *out_public_key, void *out_private_key);

void CCryptoBoringSSLShims_ED25519_keypair_from_seed(void *out_public_key,
                                                     void *out_private_key,
                                                     const void *seed);

ECDSA_SIG *CCryptoBoringSSLShims_ECDSA_do_sign(const void *digest, size_t digest_len,
                                               const EC_KEY *eckey);

int CCryptoBoringSSLShims_ECDSA_do_verify(const void *digest, size_t digest_len,
                                          const ECDSA_SIG *sig, const EC_KEY *eckey);

void CCryptoBoringSSLShims_X25519_keypair(void *out_public_value, void *out_private_key);

void CCryptoBoringSSLShims_X25519_public_from_private(void *out_public_value,
                                                      const void *private_key);

int CCryptoBoringSSLShims_X25519(void *out_shared_key, const void *private_key,
                                 const void *peer_public_value);

ECDSA_SIG *CCryptoBoringSSLShims_ECDSA_SIG_from_bytes(const void *in, size_t in_len);

int CCryptoBoringSSLShims_ED25519_verify(const void *message, size_t message_len,
                                         const void *signature, const void *public_key);

int CCryptoBoringSSLShims_ED25519_sign(void *out_sig, const void *message,
                                       size_t message_len, const void *private_key);

BIGNUM *CCryptoBoringSSLShims_BN_bin2bn(const void *in, size_t len, BIGNUM *ret);

size_t CCryptoBoringSSLShims_BN_bn2bin(const BIGNUM *in, void *out);

int CCryptoBoringSSLShims_BN_mod(BIGNUM *rem, const BIGNUM *a, const BIGNUM *m, BN_CTX *ctx);

int CCryptoBoringSSLShims_RSA_verify(int hash_nid, const void *msg, size_t msg_len,
                                     const void *sig, size_t sig_len, RSA *rsa);

int CCryptoBoringSSLShims_RSA_verify_pss_mgf1(RSA *rsa, const void *msg,
                                              size_t msg_len, const EVP_MD *md,
                                              const EVP_MD *mgf1_md, int salt_len,
                                              const void *sig, size_t sig_len);

int CCryptoBoringSSLShims_RSA_sign(int hash_nid, const void *in,
                                   unsigned int in_len, void *out,
                                   unsigned int *out_len, RSA *rsa);

int CCryptoBoringSSLShims_RSA_sign_pss_mgf1(RSA *rsa, size_t *out_len, void *out,
                                            size_t max_out, const void *in,
                                            size_t in_len, const EVP_MD *md,
                                            const EVP_MD *mgf1_md, int salt_len);

int CCryptoBoringSSLShims_RSA_public_encrypt(int flen, const void *from, void *to,
                                             RSA *rsa, int padding);

int CCryptoBoringSSLShims_RSA_private_decrypt(int flen, const void *from, void *to,
                                              RSA *rsa, int padding);

int CCryptoBoringSSLShims_EVP_PKEY_encrypt(EVP_PKEY_CTX *ctx, void *out,
                                           size_t *out_len, const void *in,
                                           size_t in_len);

int CCryptoBoringSSLShims_EVP_PKEY_decrypt(EVP_PKEY_CTX *ctx, void *out,
                                           size_t *out_len, const void *in,
                                           size_t in_len);

int CCryptoBoringSSLShims_EC_hash_to_curve_p256_xmd_sha256_sswu(const EC_GROUP *group, EC_POINT *out,
                                                                const void *dst, size_t dst_len,
                                                                const void *msg, size_t msg_len);

int CCryptoBoringSSLShims_EC_hash_to_curve_p384_xmd_sha384_sswu(const EC_GROUP *group, EC_POINT *out,
                                                                const void *dst, size_t dst_len,
                                                                const void *msg, size_t msg_len);

size_t CCryptoBoringSSLShims_EC_POINT_point2oct(const EC_GROUP *group,
                                                const EC_POINT *point,
                                                point_conversion_form_t form,
                                                void *buf, size_t max_out,
                                                BN_CTX *ctx);

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif  // C_CRYPTO_BORINGSSL_SHIMS_H
