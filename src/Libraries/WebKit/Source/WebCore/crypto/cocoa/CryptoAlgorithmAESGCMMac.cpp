/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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
#include "CryptoAlgorithmAESGCM.h"

#include "CommonCryptoUtilities.h"
#include "CryptoAlgorithmAesGcmParams.h"
#include "CryptoKeyAES.h"
#include <wtf/CryptographicUtilities.h>
#include <wtf/StdLibExtras.h>

#if HAVE(SWIFT_CPP_INTEROP)
#include <pal/PALSwift.h>
#endif

namespace WebCore {

static ExceptionOr<Vector<uint8_t>> encryptAESGCM(const Vector<uint8_t>& iv, const Vector<uint8_t>& key, const Vector<uint8_t>& plainText, const Vector<uint8_t>& additionalData, size_t desiredTagLengthInBytes)
{
    Vector<uint8_t> cipherText(plainText.size() + desiredTagLengthInBytes); // Per section 5.2.1.2: http://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf
    Vector<uint8_t> tag(desiredTagLengthInBytes);
    // tagLength is actual an input <rdar://problem/30660074>
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CCCryptorStatus status = CCCryptorGCM(kCCEncrypt, kCCAlgorithmAES, key.data(), key.size(), iv.data(), iv.size(), additionalData.data(), additionalData.size(), plainText.data(), plainText.size(), cipherText.data(), tag.data(), &desiredTagLengthInBytes);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (status)
        return Exception { ExceptionCode::OperationError };
    memcpySpan(cipherText.mutableSpan().subspan(plainText.size()), tag.span());

    return WTFMove(cipherText);
}

#if HAVE(SWIFT_CPP_INTEROP)
static ExceptionOr<Vector<uint8_t>> encryptCryptoKitAESGCM(const Vector<uint8_t>& iv, const Vector<uint8_t>& key, const Vector<uint8_t>& plainText, const Vector<uint8_t>& additionalData, size_t desiredTagLengthInBytes)
{
    auto rv = PAL::AesGcm::encrypt(key.span(), iv.span(), additionalData.span(), plainText.span(), desiredTagLengthInBytes);
    if (rv.errorCode != Cpp::ErrorCodes::Success)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(rv.result);
}
#endif

static ExceptionOr<Vector<uint8_t>> decyptAESGCM(const Vector<uint8_t>& iv, const Vector<uint8_t>& key, const Vector<uint8_t>& cipherText, const Vector<uint8_t>& additionalData, size_t desiredTagLengthInBytes)
{
    Vector<uint8_t> plainText(cipherText.size()); // Per section 5.2.1.2: http://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf
    Vector<uint8_t> tag(desiredTagLengthInBytes);
    size_t offset = cipherText.size() - desiredTagLengthInBytes;
    // tagLength is actual an input <rdar://problem/30660074>
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CCCryptorStatus status = CCCryptorGCM(kCCDecrypt, kCCAlgorithmAES, key.data(), key.size(), iv.data(), iv.size(), additionalData.data(), additionalData.size(), cipherText.data(), offset, plainText.data(), tag.data(), &desiredTagLengthInBytes);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (status)
        return Exception { ExceptionCode::OperationError };

    // Using a constant time comparison to prevent timing attacks.
    if (constantTimeMemcmp(tag.span(), cipherText.subspan(offset)))
        return Exception { ExceptionCode::OperationError };

    plainText.shrink(offset);
    return WTFMove(plainText);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESGCM::platformEncrypt(const CryptoAlgorithmAesGcmParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& plainText)
{
#if HAVE(SWIFT_CPP_INTEROP)
    if (parameters.ivVector().size() >= 12)
        return encryptCryptoKitAESGCM(parameters.ivVector(), key.key(), plainText, parameters.additionalDataVector(), parameters.tagLength.value_or(0) / 8);
    return encryptAESGCM(parameters.ivVector(), key.key(), plainText, parameters.additionalDataVector(), parameters.tagLength.value_or(0) / 8);

#else
    return encryptAESGCM(parameters.ivVector(), key.key(), plainText, parameters.additionalDataVector(), parameters.tagLength.value_or(0) / 8);
#endif

}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESGCM::platformDecrypt(const CryptoAlgorithmAesGcmParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& cipherText)
{
    // FIXME: Add decrypt with CryptoKit once rdar://92701544 is resolved.
    return decyptAESGCM(parameters.ivVector(), key.key(), cipherText, parameters.additionalDataVector(), parameters.tagLength.value_or(0) / 8);
}

} // namespace WebCore
