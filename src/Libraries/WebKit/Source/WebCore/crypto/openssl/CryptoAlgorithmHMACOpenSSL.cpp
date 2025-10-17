/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#include "CryptoAlgorithmHMAC.h"

#include "CryptoKeyHMAC.h"
#include "OpenSSLCryptoUniquePtr.h"
#include "OpenSSLUtilities.h"
#include <openssl/evp.h>
#include <wtf/CryptographicUtilities.h>

namespace WebCore {

static std::optional<Vector<uint8_t>> calculateSignature(const EVP_MD* algorithm, const Vector<uint8_t>& key, const uint8_t* data, size_t dataLength)
{
    HMACCtxPtr ctx;
    if (!(ctx = HMACCtxPtr(HMAC_CTX_new())))
        return std::nullopt;

    if (1 != HMAC_Init_ex(ctx.get(), key.data(), key.size(), algorithm, nullptr))
        return std::nullopt;

    // Call update with the message
    if (1 != HMAC_Update(ctx.get(), data, dataLength))
        return std::nullopt;

    // Finalize the DigestSign operation
    Vector<uint8_t> cipherText(EVP_MAX_MD_SIZE);
    unsigned len = 0;
    if (1 != HMAC_Final(ctx.get(), cipherText.data(), &len))
        return std::nullopt;

    cipherText.shrink(len);
    return cipherText;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmHMAC::platformSign(const CryptoKeyHMAC& key, const Vector<uint8_t>& data)
{
    auto algorithm = digestAlgorithm(key.hashAlgorithmIdentifier());
    if (!algorithm)
        return Exception { ExceptionCode::OperationError };

    auto result = calculateSignature(algorithm, key.key(), data.data(), data.size());
    if (!result)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*result);
}

ExceptionOr<bool> CryptoAlgorithmHMAC::platformVerify(const CryptoKeyHMAC& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    auto algorithm = digestAlgorithm(key.hashAlgorithmIdentifier());
    if (!algorithm)
        return Exception { ExceptionCode::OperationError };

    auto expectedSignature = calculateSignature(algorithm, key.key(), data.data(), data.size());
    if (!expectedSignature)
        return Exception { ExceptionCode::OperationError };
    // Using a constant time comparison to prevent timing attacks.
    return signature.size() == expectedSignature->size() && !constantTimeMemcmp(expectedSignature->span(), signature.span());
}

} // namespace WebCore
