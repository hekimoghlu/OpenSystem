/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#include "config.h"
#include "CryptoAlgorithmRSA_OAEP.h"

#include "CryptoAlgorithmRsaOaepParams.h"
#include "CryptoKeyRSA.h"
#include "OpenSSLUtilities.h"

namespace WebCore {

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSA_OAEP::platformEncrypt(const CryptoAlgorithmRsaOaepParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& plainText)
{
#if defined(EVP_PKEY_CTX_set_rsa_oaep_md) && defined(EVP_PKEY_CTX_set_rsa_mgf1_md) && defined(EVP_PKEY_CTX_set0_rsa_oaep_label)
    const EVP_MD* md = digestAlgorithm(key.hashAlgorithmIdentifier());
    if (!md)
        return Exception { ExceptionCode::NotSupportedError };

    auto ctx = EvpPKeyCtxPtr(EVP_PKEY_CTX_new(key.platformKey(), nullptr));
    if (!ctx)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_encrypt_init(ctx.get()) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_padding(ctx.get(), RSA_PKCS1_OAEP_PADDING) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_oaep_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_mgf1_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (!parameters.labelVector().isEmpty()) {
        size_t labelSize = parameters.labelVector().size();
        // The library takes ownership of the label so the caller should not free the original memory pointed to by label.
        auto label = OPENSSL_malloc(labelSize);
        memcpy(label, parameters.labelVector().data(), labelSize);
        if (EVP_PKEY_CTX_set0_rsa_oaep_label(ctx.get(), label, labelSize) <= 0) {
            OPENSSL_free(label);
            return Exception { ExceptionCode::OperationError };
        }
    }

    size_t cipherTextLen;
    if (EVP_PKEY_encrypt(ctx.get(), nullptr, &cipherTextLen, plainText.data(), plainText.size()) <= 0)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> cipherText(cipherTextLen);
    if (EVP_PKEY_encrypt(ctx.get(), cipherText.data(), &cipherTextLen, plainText.data(), plainText.size()) <= 0)
        return Exception { ExceptionCode::OperationError };
    cipherText.shrink(cipherTextLen);

    return cipherText;
#else
    return Exception { ExceptionCode::NotSupportedError };
#endif
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSA_OAEP::platformDecrypt(const CryptoAlgorithmRsaOaepParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& cipherText)
{
#if defined(EVP_PKEY_CTX_set_rsa_oaep_md) && defined(EVP_PKEY_CTX_set_rsa_mgf1_md) && defined(EVP_PKEY_CTX_set0_rsa_oaep_label)
    const EVP_MD* md = digestAlgorithm(key.hashAlgorithmIdentifier());
    if (!md)
        return Exception { ExceptionCode::NotSupportedError };

    auto ctx = EvpPKeyCtxPtr(EVP_PKEY_CTX_new(key.platformKey(), nullptr));
    if (!ctx)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_decrypt_init(ctx.get()) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_padding(ctx.get(), RSA_PKCS1_OAEP_PADDING) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_oaep_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_mgf1_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (!parameters.labelVector().isEmpty()) {
        size_t labelSize = parameters.labelVector().size();
        // The library takes ownership of the label so the caller should not free the original memory pointed to by label.
        auto label = OPENSSL_malloc(labelSize);
        memcpy(label, parameters.labelVector().data(), labelSize);
        if (EVP_PKEY_CTX_set0_rsa_oaep_label(ctx.get(), label, labelSize) <= 0) {
            OPENSSL_free(label);
            return Exception { ExceptionCode::OperationError };
        }
    }

    size_t plainTextLen;
    if (EVP_PKEY_decrypt(ctx.get(), nullptr, &plainTextLen, cipherText.data(), cipherText.size()) <= 0)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> plainText(plainTextLen);
    if (EVP_PKEY_decrypt(ctx.get(), plainText.data(), &plainTextLen, cipherText.data(), cipherText.size()) <= 0)
        return Exception { ExceptionCode::OperationError };
    plainText.shrink(plainTextLen);

    return plainText;
#else
    return Exception { ExceptionCode::NotSupportedError };
#endif
}

} // namespace WebCore
