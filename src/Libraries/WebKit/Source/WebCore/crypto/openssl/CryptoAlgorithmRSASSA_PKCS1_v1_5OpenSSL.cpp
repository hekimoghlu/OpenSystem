/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#include "CryptoAlgorithmRSASSA_PKCS1_v1_5.h"

#include "CryptoKeyRSA.h"
#include "OpenSSLUtilities.h"

namespace WebCore {

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSASSA_PKCS1_v1_5::platformSign(const CryptoKeyRSA& key, const Vector<uint8_t>& data)
{
    const EVP_MD* md = digestAlgorithm(key.hashAlgorithmIdentifier());
    if (!md)
        return Exception { ExceptionCode::NotSupportedError };

    std::optional<Vector<uint8_t>> digest = calculateDigest(md, data);
    if (!digest)
        return Exception { ExceptionCode::OperationError };

    auto ctx = EvpPKeyCtxPtr(EVP_PKEY_CTX_new(key.platformKey(), nullptr));
    if (!ctx)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_sign_init(ctx.get()) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_padding(ctx.get(), RSA_PKCS1_PADDING) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_signature_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    size_t signatureLen;
    if (EVP_PKEY_sign(ctx.get(), nullptr, &signatureLen, digest->data(), digest->size()) <= 0)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> signature(signatureLen);
    if (EVP_PKEY_sign(ctx.get(), signature.data(), &signatureLen, digest->data(), digest->size()) <= 0)
        return Exception { ExceptionCode::OperationError };
    signature.shrink(signatureLen);

    return signature;
}

ExceptionOr<bool> CryptoAlgorithmRSASSA_PKCS1_v1_5::platformVerify(const CryptoKeyRSA& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    const EVP_MD* md = digestAlgorithm(key.hashAlgorithmIdentifier());
    if (!md)
        return Exception { ExceptionCode::NotSupportedError };

    std::optional<Vector<uint8_t>> digest = calculateDigest(md, data);
    if (!digest)
        return Exception { ExceptionCode::OperationError };

    auto ctx = EvpPKeyCtxPtr(EVP_PKEY_CTX_new(key.platformKey(), nullptr));
    if (!ctx)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_verify_init(ctx.get()) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_padding(ctx.get(), RSA_PKCS1_PADDING) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_signature_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    int ret = EVP_PKEY_verify(ctx.get(), signature.data(), signature.size(), digest->data(), digest->size());

    return ret == 1;
}

} // namespace WebCore
