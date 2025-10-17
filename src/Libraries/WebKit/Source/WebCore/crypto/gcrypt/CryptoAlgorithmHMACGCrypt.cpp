/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
#include <pal/crypto/gcrypt/Handle.h>
#include <wtf/CryptographicUtilities.h>

namespace WebCore {

static int getGCryptDigestAlgorithm(CryptoAlgorithmIdentifier hashFunction)
{
    switch (hashFunction) {
    case CryptoAlgorithmIdentifier::SHA_1:
        return GCRY_MAC_HMAC_SHA1;
    case CryptoAlgorithmIdentifier::SHA_224:
        return GCRY_MAC_HMAC_SHA224;
    case CryptoAlgorithmIdentifier::SHA_256:
        return GCRY_MAC_HMAC_SHA256;
    case CryptoAlgorithmIdentifier::SHA_384:
        return GCRY_MAC_HMAC_SHA384;
    case CryptoAlgorithmIdentifier::SHA_512:
        return GCRY_MAC_HMAC_SHA512;
    default:
        return GCRY_MAC_NONE;
    }
}

static std::optional<Vector<uint8_t>> calculateSignature(int algorithm, const Vector<uint8_t>& key, const uint8_t* data, size_t dataLength)
{
    const void* keyData = key.data() ? key.data() : reinterpret_cast<const uint8_t*>("");

    PAL::GCrypt::Handle<gcry_mac_hd_t> hd;
    gcry_error_t err = gcry_mac_open(&hd, algorithm, 0, nullptr);
    if (err)
        return std::nullopt;

    err = gcry_mac_setkey(hd, keyData, key.size());
    if (err)
        return std::nullopt;

    err = gcry_mac_write(hd, data, dataLength);
    if (err)
        return std::nullopt;

    size_t digestLength = gcry_mac_get_algo_maclen(algorithm);
    Vector<uint8_t> signature(digestLength);
    err = gcry_mac_read(hd, signature.data(), &digestLength);
    if (err)
        return std::nullopt;

    signature.resize(digestLength);
    return signature;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmHMAC::platformSign(const CryptoKeyHMAC& key, const Vector<uint8_t>& data)
{
    auto algorithm = getGCryptDigestAlgorithm(key.hashAlgorithmIdentifier());
    if (algorithm == GCRY_MAC_NONE)
        return Exception { ExceptionCode::OperationError };

    auto result = calculateSignature(algorithm, key.key(), data.data(), data.size());
    if (!result)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*result);
}

ExceptionOr<bool> CryptoAlgorithmHMAC::platformVerify(const CryptoKeyHMAC& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    auto algorithm = getGCryptDigestAlgorithm(key.hashAlgorithmIdentifier());
    if (algorithm == GCRY_MAC_NONE)
        return Exception { ExceptionCode::OperationError };

    auto expectedSignature = calculateSignature(algorithm, key.key(), data.data(), data.size());
    if (!expectedSignature)
        return Exception { ExceptionCode::OperationError };
    // Using a constant time comparison to prevent timing attacks.
    return signature.size() == expectedSignature->size() && !constantTimeMemcmp(expectedSignature->span(), signature.span());
}

} // namespace WebCore
