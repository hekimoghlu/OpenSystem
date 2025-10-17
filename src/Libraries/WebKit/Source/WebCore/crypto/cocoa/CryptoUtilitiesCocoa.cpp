/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#include "CryptoUtilitiesCocoa.h"

#include "CryptoAlgorithmAESCTR.h"
#include <CommonCrypto/CommonCrypto.h>
#include <pal/spi/cocoa/CommonCryptoSPI.h>

namespace WebCore {

ExceptionOr<Vector<uint8_t>> transformAESCTR(CCOperation operation, const Vector<uint8_t>& counter, size_t counterLength, const Vector<uint8_t>& key, std::span<const uint8_t> data)
{
    // FIXME: We should remove the following hack once <rdar://problem/31361050> is fixed.
    // counter = nonce + counter
    // CommonCrypto currently can neither reset the counter nor detect overflow once the counter reaches its max value restricted
    // by the counterLength. It then increments the nonce which should stay same for the whole operation. To remedy this issue,
    // we detect the overflow ahead and divide the operation into two parts.
    size_t numberOfBlocks = data.size() % kCCBlockSizeAES128 ? data.size() / kCCBlockSizeAES128 + 1 : data.size() / kCCBlockSizeAES128;

    // Detect loop
    if (counterLength < sizeof(size_t) * 8 && numberOfBlocks > (static_cast<size_t>(1) << counterLength))
        return Exception { ExceptionCode::OperationError };

    // Calculate capacity before overflow
    CryptoAlgorithmAESCTR::CounterBlockHelper counterBlockHelper(counter, counterLength);
    size_t capacity = counterBlockHelper.countToOverflowSaturating();

    // Divide data into two parts if necessary.
    size_t headSize = data.size();
    if (capacity < numberOfBlocks)
        headSize = capacity * kCCBlockSizeAES128;

    // first part: compute the first n=capacity blocks of data if capacity is insufficient. Otherwise, return the result.
    CCCryptorRef cryptor;
    CCCryptorStatus status = CCCryptorCreateWithMode(operation, kCCModeCTR, kCCAlgorithmAES128, ccNoPadding, counter.data(), key.data(), key.size(), 0, 0, 0, kCCModeOptionCTR_BE, &cryptor);
    if (status)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> head(CCCryptorGetOutputLength(cryptor, headSize, true));

    size_t bytesWritten;
    status = CCCryptorUpdate(cryptor, data.data(), headSize, head.data(), head.size(), &bytesWritten);
    if (status)
        return Exception { ExceptionCode::OperationError };

    auto p = head.mutableSpan().subspan(bytesWritten);
    status = CCCryptorFinal(cryptor, p.data(), p.size(), &bytesWritten);
    skip(p, bytesWritten);
    if (status)
        return Exception { ExceptionCode::OperationError };

    head.shrink(head.size() - p.size());

    CCCryptorRelease(cryptor);

    if (capacity >= numberOfBlocks)
        return WTFMove(head);

    // second part: compute the remaining data and append them to the head.
    // reset counter
    Vector<uint8_t> remainingCounter = counterBlockHelper.counterVectorAfterOverflow();
    status = CCCryptorCreateWithMode(operation, kCCModeCTR, kCCAlgorithmAES128, ccNoPadding, remainingCounter.data(), key.data(), key.size(), 0, 0, 0, kCCModeOptionCTR_BE, &cryptor);
    if (status)
        return Exception { ExceptionCode::OperationError };

    size_t tailSize = data.size() - headSize;
    Vector<uint8_t> tail(CCCryptorGetOutputLength(cryptor, tailSize, true));

    auto dataAfterHeader = data.subspan(headSize);
    status = CCCryptorUpdate(cryptor, dataAfterHeader.data(), dataAfterHeader.size(), tail.data(), tail.size(), &bytesWritten);
    if (status)
        return Exception { ExceptionCode::OperationError };

    p = tail.mutableSpan().subspan(bytesWritten);
    status = CCCryptorFinal(cryptor, p.data(), p.size(), &bytesWritten);
    skip(p, bytesWritten);
    if (status)
        return Exception { ExceptionCode::OperationError };

    tail.shrink(tail.size() - p.size());

    CCCryptorRelease(cryptor);

    head.appendVector(tail);
    return WTFMove(head);
}

CCStatus keyDerivationHMAC(CCDigestAlgorithm digest, std::span<const uint8_t> keyDerivationKey, std::span<const uint8_t> context, std::span<const uint8_t> salt, Vector<uint8_t>& derivedKey)
{
    CCKDFParametersRef params;
    CCStatus rv = CCKDFParametersCreateHkdf(&params, salt.data(), salt.size(), context.data(), context.size());
    if (rv != kCCSuccess)
        return rv;

    rv = CCDeriveKey(params, digest, keyDerivationKey.data(), keyDerivationKey.size(), derivedKey.data(), derivedKey.size());
    CCKDFParametersDestroy(params);

    return rv;
}

ExceptionOr<Vector<uint8_t>> deriveHDKFBits(CCDigestAlgorithm digestAlgorithm, std::span<const uint8_t> key, std::span<const uint8_t> salt, std::span<const uint8_t> info, size_t length)
{
    Vector<uint8_t> result(length / 8);

    // <rdar://problem/32439455> Currently, when key data is empty, CCKeyDerivationHMac will bail out.
    if (keyDerivationHMAC(digestAlgorithm, key, info, salt, result) != kCCSuccess)
        return Exception { ExceptionCode::OperationError };

    return WTFMove(result);
}

ExceptionOr<Vector<uint8_t>> deriveHDKFSHA256Bits(std::span<const uint8_t> key, std::span<const uint8_t> salt, std::span<const uint8_t> info, size_t length)
{
    return deriveHDKFBits(kCCDigestSHA256, key, salt, info, length);
}

Vector<uint8_t> calculateHMACSignature(CCHmacAlgorithm algorithm, const Vector<uint8_t>& key, std::span<const uint8_t> data)
{
    size_t digestLength;
    switch (algorithm) {
    case kCCHmacAlgSHA1:
        digestLength = CC_SHA1_DIGEST_LENGTH;
        break;
    case kCCHmacAlgSHA224:
        digestLength = CC_SHA224_DIGEST_LENGTH;
        break;
    case kCCHmacAlgSHA256:
        digestLength = CC_SHA256_DIGEST_LENGTH;
        break;
    case kCCHmacAlgSHA384:
        digestLength = CC_SHA384_DIGEST_LENGTH;
        break;
    case kCCHmacAlgSHA512:
        digestLength = CC_SHA512_DIGEST_LENGTH;
        break;
    default:
        ASSERT_NOT_REACHED();
        return Vector<uint8_t>();
    }

    Vector<uint8_t> result(digestLength);
    CCHmac(algorithm, key.data(), key.size(), data.data(), data.size(), result.data());
    return result;
}

Vector<uint8_t> calculateSHA256Signature(const Vector<uint8_t>& key, std::span<const uint8_t> data)
{
    return calculateHMACSignature(kCCHmacAlgSHA256, key, data);
}

} // namespace WebCore
