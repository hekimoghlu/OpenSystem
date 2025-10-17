/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#include "CryptoAlgorithmRSA_PSS.h"

#if HAVE(RSA_PSS)

#include "CryptoAlgorithmRsaPssParams.h"
#include "CryptoKeyRSA.h"
#include "OpenSSLUtilities.h"

namespace WebCore {

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSA_PSS::platformSign(const CryptoAlgorithmRsaPssParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& data)
{
#if defined(EVP_PKEY_CTX_set_rsa_pss_saltlen) && defined(EVP_PKEY_CTX_set_rsa_mgf1_md)
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

    if (EVP_PKEY_CTX_set_rsa_padding(ctx.get(), RSA_PKCS1_PSS_PADDING ) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_pss_saltlen(ctx.get(), parameters.saltLength) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_signature_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_mgf1_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    size_t signatureLen;
    if (EVP_PKEY_sign(ctx.get(), nullptr, &signatureLen, digest->data(), digest->size()) <= 0)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> signature(signatureLen);
    if (EVP_PKEY_sign(ctx.get(), signature.data(), &signatureLen, digest->data(), digest->size()) <= 0)
        return Exception { ExceptionCode::OperationError };
    signature.shrink(signatureLen);

    return signature;
#else
    return Exception { ExceptionCode::NotSupportedError };
#endif
}

ExceptionOr<bool> CryptoAlgorithmRSA_PSS::platformVerify(const CryptoAlgorithmRsaPssParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
#if defined(EVP_PKEY_CTX_set_rsa_pss_saltlen) && defined(EVP_PKEY_CTX_set_rsa_mgf1_md)
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

    if (EVP_PKEY_CTX_set_rsa_padding(ctx.get(), RSA_PKCS1_PSS_PADDING ) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_pss_saltlen(ctx.get(), parameters.saltLength) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_signature_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    if (EVP_PKEY_CTX_set_rsa_mgf1_md(ctx.get(), md) <= 0)
        return Exception { ExceptionCode::OperationError };

    int ret = EVP_PKEY_verify(ctx.get(), signature.data(), signature.size(), digest->data(), digest->size());

    return ret == 1;
#else
    return Exception { ExceptionCode::NotSupportedError };
#endif
}

} // namespace WebCore

#endif // HAVE(RSA_PSS)
