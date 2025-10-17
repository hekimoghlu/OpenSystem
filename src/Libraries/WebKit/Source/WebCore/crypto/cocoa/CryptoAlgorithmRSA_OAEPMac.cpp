/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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

#include "CommonCryptoUtilities.h"
#include "CryptoAlgorithmRsaOaepParams.h"
#include "CryptoKeyRSA.h"

namespace WebCore {

static ExceptionOr<Vector<uint8_t>> encryptRSA_OAEP(CryptoAlgorithmIdentifier hash, const Vector<uint8_t>& label, const PlatformRSAKey key, size_t keyLength, const Vector<uint8_t>& data)
{
    CCDigestAlgorithm digestAlgorithm;
    if (!getCommonCryptoDigestAlgorithm(hash, digestAlgorithm))
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> cipherText(keyLength / 8); // Per Step 3.c of https://tools.ietf.org/html/rfc3447#section-7.1.1
    size_t cipherTextLength = cipherText.size();
    if (CCRSACryptorEncrypt(key, ccOAEPPadding, data.data(), data.size(), cipherText.data(), &cipherTextLength, label.data(), label.size(), digestAlgorithm))
        return Exception { ExceptionCode::OperationError };

    return WTFMove(cipherText);
}

static ExceptionOr<Vector<uint8_t>> decryptRSA_OAEP(CryptoAlgorithmIdentifier hash, const Vector<uint8_t>& label, const PlatformRSAKey key, size_t keyLength, const Vector<uint8_t>& data)
{
    CCDigestAlgorithm digestAlgorithm;
    if (!getCommonCryptoDigestAlgorithm(hash, digestAlgorithm))
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> plainText(keyLength / 8); // Per Step 1.b of https://tools.ietf.org/html/rfc3447#section-7.1.1
    size_t plainTextLength = plainText.size();
    if (CCRSACryptorDecrypt(key, ccOAEPPadding, data.data(), data.size(), plainText.data(), &plainTextLength, label.data(), label.size(), digestAlgorithm))
        return Exception { ExceptionCode::OperationError };

    plainText.resize(plainTextLength);
    return WTFMove(plainText);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSA_OAEP::platformEncrypt(const CryptoAlgorithmRsaOaepParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& plainText)
{
    return encryptRSA_OAEP(key.hashAlgorithmIdentifier(), parameters.labelVector(), key.platformKey(), key.keySizeInBits(), plainText);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSA_OAEP::platformDecrypt(const CryptoAlgorithmRsaOaepParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& cipherText)
{
    return decryptRSA_OAEP(key.hashAlgorithmIdentifier(), parameters.labelVector(), key.platformKey(), key.keySizeInBits(), cipherText);
}

} // namespace WebCore
