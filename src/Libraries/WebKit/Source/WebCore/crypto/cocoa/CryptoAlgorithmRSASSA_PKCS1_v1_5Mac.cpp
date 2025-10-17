/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#include "CommonCryptoUtilities.h"
#include "CryptoDigestAlgorithm.h"
#include "CryptoKeyRSA.h"

namespace WebCore {

static ExceptionOr<Vector<uint8_t>> signRSASSA_PKCS1_v1_5(CryptoAlgorithmIdentifier hash, const PlatformRSAKey key, size_t keyLength, const Vector<uint8_t>& data)
{
    CCDigestAlgorithm digestAlgorithm;
    if (!getCommonCryptoDigestAlgorithm(hash, digestAlgorithm))
        return Exception { ExceptionCode::OperationError };

    auto cryptoDigestAlgorithm = WebCore::cryptoDigestAlgorithm(hash);
    if (!cryptoDigestAlgorithm)
        return Exception { ExceptionCode::OperationError };
    auto digest = PAL::CryptoDigest::create(*cryptoDigestAlgorithm);
    if (!digest)
        return Exception { ExceptionCode::OperationError };
    digest->addBytes(data.span());
    auto digestData = digest->computeHash();

    Vector<uint8_t> signature(keyLength / 8); // Per https://tools.ietf.org/html/rfc3447#section-8.2.1
    size_t signatureSize = signature.size();

    CCCryptorStatus status = CCRSACryptorSign(key, ccPKCS1Padding, digestData.data(), digestData.size(), digestAlgorithm, 0, signature.data(), &signatureSize);
    if (status)
        return Exception { ExceptionCode::OperationError };

    return WTFMove(signature);
}

static ExceptionOr<bool> verifyRSASSA_PKCS1_v1_5(CryptoAlgorithmIdentifier hash, const PlatformRSAKey key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    CCDigestAlgorithm digestAlgorithm;
    if (!getCommonCryptoDigestAlgorithm(hash, digestAlgorithm))
        return Exception { ExceptionCode::OperationError };

    auto cryptoDigestAlgorithm = WebCore::cryptoDigestAlgorithm(hash);
    if (!cryptoDigestAlgorithm)
        return Exception { ExceptionCode::OperationError };
    auto digest = PAL::CryptoDigest::create(*cryptoDigestAlgorithm);
    if (!digest)
        return Exception { ExceptionCode::OperationError };
    digest->addBytes(data.span());
    auto digestData = digest->computeHash();

    auto status = CCRSACryptorVerify(key, ccPKCS1Padding, digestData.data(), digestData.size(), digestAlgorithm, 0, signature.data(), signature.size());
    if (!status)
        return true;
    if (status == kCCDecodeError)
        return false;

    return Exception { ExceptionCode::OperationError };
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSASSA_PKCS1_v1_5::platformSign(const CryptoKeyRSA& key, const Vector<uint8_t>& data)
{
    return signRSASSA_PKCS1_v1_5(key.hashAlgorithmIdentifier(), key.platformKey(), key.keySizeInBits(), data);
}

ExceptionOr<bool> CryptoAlgorithmRSASSA_PKCS1_v1_5::platformVerify(const CryptoKeyRSA& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    return verifyRSASSA_PKCS1_v1_5(key.hashAlgorithmIdentifier(), key.platformKey(), signature, data);
}

} // namespace WebCore
