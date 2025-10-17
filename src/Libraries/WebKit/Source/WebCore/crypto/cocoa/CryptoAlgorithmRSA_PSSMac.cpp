/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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

#include "CommonCryptoUtilities.h"
#include "CryptoAlgorithmRsaPssParams.h"
#include "CryptoDigestAlgorithm.h"
#include "CryptoKeyRSA.h"

namespace WebCore {

static ExceptionOr<Vector<uint8_t>> signRSA_PSS(CryptoAlgorithmIdentifier hash, const PlatformRSAKey key, size_t keyLength, const Vector<uint8_t>& data, size_t saltLength)
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

    Vector<uint8_t> signature(keyLength / 8); // Per https://tools.ietf.org/html/rfc3447#section-8.1.1
    size_t signatureSize = signature.size();

    CCCryptorStatus status = CCRSACryptorSign(key, ccRSAPSSPadding, digestData.data(), digestData.size(), digestAlgorithm, saltLength, signature.data(), &signatureSize);
    if (status)
        return Exception { ExceptionCode::OperationError };

    return WTFMove(signature);
}

static ExceptionOr<bool> verifyRSA_PSS(CryptoAlgorithmIdentifier hash, const PlatformRSAKey key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data, size_t saltLength)
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

    auto status = CCRSACryptorVerify(key, ccRSAPSSPadding, digestData.data(), digestData.size(), digestAlgorithm, saltLength, signature.data(), signature.size());
    if (!status)
        return true;
    return false;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSA_PSS::platformSign(const CryptoAlgorithmRsaPssParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& data)
{
    return signRSA_PSS(key.hashAlgorithmIdentifier(), key.platformKey(), key.keySizeInBits(), data, parameters.saltLength);
}

ExceptionOr<bool> CryptoAlgorithmRSA_PSS::platformVerify(const CryptoAlgorithmRsaPssParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    return verifyRSA_PSS(key.hashAlgorithmIdentifier(), key.platformKey(), signature, data, parameters.saltLength);
}

} // namespace WebCore

#endif // HAVE(RSA_PSS)
