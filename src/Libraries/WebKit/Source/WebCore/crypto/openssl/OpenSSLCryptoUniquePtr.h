/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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
#pragma once

#include <memory>
#include <openssl/ec.h>
#include <openssl/hmac.h>
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
#include <openssl/kdf.h>
#include <openssl/param_build.h>
#endif
#include <openssl/x509.h>

namespace WebCore {

template <typename T>
struct OpenSSLCryptoPtrDeleter {
    void operator()(T* ptr) const = delete;
};

template <typename T>
using OpenSSLCryptoPtr = std::unique_ptr<T, OpenSSLCryptoPtrDeleter<T>>;

#define DEFINE_OPENSSL_CRYPTO_PTR_FULL(alias, typeName, deleterFunc) \
    template<> struct OpenSSLCryptoPtrDeleter<typeName> { \
        void operator()(typeName* ptr) const { \
            deleterFunc;                                             \
        }                                                            \
    };                                                               \
    using alias = OpenSSLCryptoPtr<typeName>;

#define DEFINE_OPENSSL_CRYPTO_PTR(alias, typeName, deleterFunc)      \
    DEFINE_OPENSSL_CRYPTO_PTR_FULL(alias, typeName, deleterFunc(ptr))

DEFINE_OPENSSL_CRYPTO_PTR(EvpCipherCtxPtr, EVP_CIPHER_CTX, EVP_CIPHER_CTX_free)
DEFINE_OPENSSL_CRYPTO_PTR(EvpDigestCtxPtr, EVP_MD_CTX, EVP_MD_CTX_free)
DEFINE_OPENSSL_CRYPTO_PTR(EvpPKeyPtr, EVP_PKEY, EVP_PKEY_free)
DEFINE_OPENSSL_CRYPTO_PTR(EvpPKeyCtxPtr, EVP_PKEY_CTX, EVP_PKEY_CTX_free)

#if OPENSSL_VERSION_NUMBER >= 0x30000000L
DEFINE_OPENSSL_CRYPTO_PTR(OsslParamBldPtr, OSSL_PARAM_BLD, OSSL_PARAM_BLD_free)
DEFINE_OPENSSL_CRYPTO_PTR(OsslParamPtr, OSSL_PARAM, OSSL_PARAM_free)
DEFINE_OPENSSL_CRYPTO_PTR(EVPKDFCtxPtr, EVP_KDF_CTX, EVP_KDF_CTX_free)
DEFINE_OPENSSL_CRYPTO_PTR(EVPKDFPtr, EVP_KDF, EVP_KDF_free)
#endif // OPENSSL_VERSION_NUMBER >= 0x30000000L

// These are deprecated in OpenSSL 3. FIXME: Migrate to EvpKey. See Bug #245146.
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
DEFINE_OPENSSL_CRYPTO_PTR(RSAPtr, RSA, RSA_free)
DEFINE_OPENSSL_CRYPTO_PTR(ECKeyPtr, EC_KEY, EC_KEY_free)
DEFINE_OPENSSL_CRYPTO_PTR(HMACCtxPtr, HMAC_CTX, HMAC_CTX_free)
ALLOW_DEPRECATED_DECLARATIONS_END

DEFINE_OPENSSL_CRYPTO_PTR(ECPointPtr, EC_POINT, EC_POINT_clear_free)
DEFINE_OPENSSL_CRYPTO_PTR(PKCS8PrivKeyInfoPtr, PKCS8_PRIV_KEY_INFO, PKCS8_PRIV_KEY_INFO_free)
DEFINE_OPENSSL_CRYPTO_PTR(BIGNUMPtr, BIGNUM, BN_clear_free)
DEFINE_OPENSSL_CRYPTO_PTR(BNCtxPtr, BN_CTX, BN_CTX_free)
DEFINE_OPENSSL_CRYPTO_PTR(ECDSASigPtr, ECDSA_SIG, ECDSA_SIG_free)
DEFINE_OPENSSL_CRYPTO_PTR(X509Ptr, X509, X509_free)
DEFINE_OPENSSL_CRYPTO_PTR(BIOPtr, BIO, BIO_free)

DEFINE_OPENSSL_CRYPTO_PTR_FULL(ASN1SequencePtr, ASN1_SEQUENCE_ANY, sk_ASN1_TYPE_pop_free(ptr, ASN1_TYPE_free))

#undef DEFINE_OPENSSL_CRYPTO_PTR
#undef DEFINE_OPENSSL_CRYPTO_PTR_FULL

} // namespace WebCore
