/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#include "CryptoUtilitiesCocoa.h"
#include <CommonCrypto/CommonHMAC.h>
#if HAVE(SWIFT_CPP_INTEROP)
#include <pal/PALSwiftUtils.h>
#endif
#include <wtf/CryptographicUtilities.h>

namespace WebCore {

static std::optional<CCHmacAlgorithm> commonCryptoHMACAlgorithm(CryptoAlgorithmIdentifier hashFunction)
{
    switch (hashFunction) {
    case CryptoAlgorithmIdentifier::SHA_1:
        return kCCHmacAlgSHA1;
    case CryptoAlgorithmIdentifier::SHA_224:
        return kCCHmacAlgSHA224;
    case CryptoAlgorithmIdentifier::SHA_256:
        return kCCHmacAlgSHA256;
    case CryptoAlgorithmIdentifier::SHA_384:
        return kCCHmacAlgSHA384;
    case CryptoAlgorithmIdentifier::SHA_512:
        return kCCHmacAlgSHA512;
    default:
        return std::nullopt;
    }
}

#if HAVE(SWIFT_CPP_INTEROP)
static ExceptionOr<Vector<uint8_t>> platformSignCryptoKit(const CryptoKeyHMAC& key, const Vector<uint8_t>& data)
{
    if (!isValidHashParameter(key.hashAlgorithmIdentifier()))
        return Exception { ExceptionCode::OperationError };
    return PAL::HMAC::sign(key.key().span(), data.span(), toCKHashFunction(key.hashAlgorithmIdentifier()));
}
static ExceptionOr<bool> platformVerifyCryptoKit(const CryptoKeyHMAC& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    if (!isValidHashParameter(key.hashAlgorithmIdentifier()))
        return Exception { ExceptionCode::OperationError };
    return PAL::HMAC::verify(signature.span(), key.key().span(), data.span(), toCKHashFunction(key.hashAlgorithmIdentifier()));
}
#endif

static ExceptionOr<Vector<uint8_t>> platformSignCC(const CryptoKeyHMAC& key, const Vector<uint8_t>& data)
{
    auto algorithm = commonCryptoHMACAlgorithm(key.hashAlgorithmIdentifier());
    if (!algorithm)
        return Exception { ExceptionCode::OperationError };

    return calculateHMACSignature(*algorithm, key.key(), data.span());
}

static ExceptionOr<bool> platformVerifyCC(const CryptoKeyHMAC& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    auto algorithm = commonCryptoHMACAlgorithm(key.hashAlgorithmIdentifier());
    if (!algorithm)
        return Exception { ExceptionCode::OperationError };

    auto expectedSignature = calculateHMACSignature(*algorithm, key.key(), data.span());
    // Using a constant time comparison to prevent timing attacks.
    return signature.size() == expectedSignature.size() && !constantTimeMemcmp(expectedSignature.span(), signature.span());
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmHMAC::platformSign(const CryptoKeyHMAC& key, const Vector<uint8_t>& data)
{
#if HAVE(SWIFT_CPP_INTEROP)
    if (key.hashAlgorithmIdentifier() != CryptoAlgorithmIdentifier::SHA_224)
        return platformSignCryptoKit(key, data);
    return platformSignCC(key, data);
#else
    return platformSignCC(key, data);
#endif
}

ExceptionOr<bool> CryptoAlgorithmHMAC::platformVerify(const CryptoKeyHMAC& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
#if HAVE(SWIFT_CPP_INTEROP)
    if (key.hashAlgorithmIdentifier() != CryptoAlgorithmIdentifier::SHA_224)
        return platformVerifyCryptoKit(key, signature, data);
    return platformVerifyCC(key, signature, data);
#else
    return platformVerifyCC(key, signature, data);
#endif
}
} // namespace WebCore
