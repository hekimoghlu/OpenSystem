/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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
#include "CryptoAlgorithmECDSA.h"

#include "CommonCryptoDERUtilities.h"
#include "CommonCryptoUtilities.h"
#include "CryptoAlgorithmEcdsaParams.h"
#include "CryptoDigestAlgorithm.h"
#include "CryptoKeyEC.h"
#include <wtf/StdLibExtras.h>

#if HAVE(SWIFT_CPP_INTEROP)
#include <pal/PALSwiftUtils.h>
#endif

namespace WebCore {
#if HAVE(SWIFT_CPP_INTEROP)

static ExceptionOr<Vector<uint8_t>> signECDSACryptoKit(CryptoAlgorithmIdentifier hash, const PlatformECKeyContainer& key, const Vector<uint8_t>& data)
{
    if (!isValidHashParameter(hash))
        return Exception { ExceptionCode::OperationError };
    auto rv = key->sign(data.span(), toCKHashFunction(hash));
    if (rv.errorCode != Cpp::ErrorCodes::Success)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(rv.result);
}

static ExceptionOr<bool> verifyECDSACryptoKit(CryptoAlgorithmIdentifier hash, const PlatformECKeyContainer& key, const Vector<uint8_t>& signature, const Vector<uint8_t> data)
{
    if (!isValidHashParameter(hash))
        return Exception { ExceptionCode::OperationError };
    return key->verify(data.span(), signature.span(), toCKHashFunction(hash)).errorCode == Cpp::ErrorCodes::Success;
}
#else

static ExceptionOr<Vector<uint8_t>> signECDSA(CryptoAlgorithmIdentifier hash, const PlatformECKeyContainer& key, size_t keyLengthInBytes, const Vector<uint8_t>& data)
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

    // The signature produced by CCECCryptorSignHash is in DER format.
    // tag + length(1) + tag + length(1) + InitialOctet(?) + keyLength in bytes + tag + length(1) + InitialOctet(?) + keyLength in bytes
    Vector<uint8_t> signature(8 + keyLengthInBytes * 2);
    size_t signatureSize = signature.size();
    CCCryptorStatus status = CCECCryptorSignHash(key.get(), digestData.data(), digestData.size(), signature.data(), &signatureSize);
    if (status)
        return Exception { ExceptionCode::OperationError };

    // FIXME: <rdar://problem/31618371>
    // convert the DER binary into r + s
    Vector<uint8_t> newSignature;
    newSignature.reserveInitialCapacity(keyLengthInBytes * 2);
    size_t offset = 1; // skip tag
    offset += bytesUsedToEncodedLength(signature[offset]); // skip length
    ++offset; // skip tag

    // If r < keyLengthInBytes, fill the head of r with 0s.
    size_t bytesToCopy = keyLengthInBytes;
    if (signature[offset] < keyLengthInBytes) {
        newSignature.grow(keyLengthInBytes - signature[offset]);
        memsetSpan(newSignature.mutableSpan().first(keyLengthInBytes - signature[offset]), InitialOctet);
        bytesToCopy = signature[offset];
    } else if (signature[offset] > keyLengthInBytes) // Otherwise skip the leading 0s of r.
        offset += signature[offset] - keyLengthInBytes;
    offset++; // skip length
    ASSERT_WITH_SECURITY_IMPLICATION(signature.size() > offset + bytesToCopy);
    newSignature.append(signature.subspan(offset, bytesToCopy));
    offset += bytesToCopy + 1; // skip r, tag

    // If s < keyLengthInBytes, fill the head of s with 0s.
    bytesToCopy = keyLengthInBytes;
    if (signature[offset] < keyLengthInBytes) {
        size_t pos = newSignature.size();
        newSignature.resize(pos + keyLengthInBytes - signature[offset]);
        memsetSpan(newSignature.mutableSpan().subspan(pos), InitialOctet);
        bytesToCopy = signature[offset];
    } else if (signature[offset] > keyLengthInBytes) // Otherwise skip the leading 0s of s.
        offset += signature[offset] - keyLengthInBytes;
    ++offset; // skip length
    ASSERT_WITH_SECURITY_IMPLICATION(signature.size() >= offset + bytesToCopy);
    newSignature.append(signature.subspan(offset, bytesToCopy));

    return WTFMove(newSignature);
}

static ExceptionOr<bool> verifyECDSA(CryptoAlgorithmIdentifier hash, const PlatformECKeyContainer& key, size_t keyLengthInBytes, const Vector<uint8_t>& signature, const Vector<uint8_t> data)
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

    if (signature.size() != keyLengthInBytes * 2)
        return false;

    // FIXME: <rdar://problem/31618371>
    // Convert the signature into DER format.
    // tag + length(1) + tag + length(1) + InitialOctet(?) + r + tag + length(1) + InitialOctet(?) + s
    // Skip any heading 0s of r and s.
    size_t rStart = 0;
    while (rStart < keyLengthInBytes && !signature[rStart])
        rStart++;
    size_t sStart = keyLengthInBytes;
    while (sStart < signature.size() && !signature[sStart])
        sStart++;
    if (rStart >= keyLengthInBytes || sStart >= signature.size())
        return false;

    // InitialOctet is needed when the first byte of r/s is larger than or equal to 128.
    bool rNeedsInitialOctet = signature[rStart] >= 128;
    bool sNeedsInitialOctet = signature[sStart] >= 128;

    // Construct the DER signature.
    Vector<uint8_t> newSignature;
    newSignature.reserveInitialCapacity(6 + keyLengthInBytes * 3  + rNeedsInitialOctet + sNeedsInitialOctet - rStart - sStart);
    newSignature.append(SequenceMark);
    addEncodedASN1Length(newSignature, 4 + keyLengthInBytes * 3  + rNeedsInitialOctet + sNeedsInitialOctet - rStart - sStart);
    newSignature.append(IntegerMark);
    addEncodedASN1Length(newSignature, keyLengthInBytes + rNeedsInitialOctet - rStart);
    if (rNeedsInitialOctet)
        newSignature.append(InitialOctet);
    newSignature.append(signature.subspan(rStart, keyLengthInBytes - rStart));
    newSignature.append(IntegerMark);
    addEncodedASN1Length(newSignature, keyLengthInBytes * 2 + sNeedsInitialOctet - sStart);
    if (sNeedsInitialOctet)
        newSignature.append(InitialOctet);
    newSignature.append(signature.subspan(sStart, keyLengthInBytes * 2 - sStart));

    uint32_t valid;
    CCCryptorStatus status = CCECCryptorVerifyHash(key.get(), digestData.data(), digestData.size(), newSignature.data(), newSignature.size(), &valid);
    if (status) {
        WTFLogAlways("ERROR: CCECCryptorVerifyHash() returns error=%d", status);
        return false;
    }
    return valid;
}
#endif

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmECDSA::platformSign(const CryptoAlgorithmEcdsaParams& parameters, const CryptoKeyEC& key, const Vector<uint8_t>& data)
{
#if HAVE(SWIFT_CPP_INTEROP)
    return signECDSACryptoKit(parameters.hashIdentifier, key.platformKey(), data);
#else
    return signECDSA(parameters.hashIdentifier, key.platformKey(), key.keySizeInBytes(), data);
#endif
}

ExceptionOr<bool> CryptoAlgorithmECDSA::platformVerify(const CryptoAlgorithmEcdsaParams& parameters, const CryptoKeyEC& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
#if HAVE(SWIFT_CPP_INTEROP)
    return verifyECDSACryptoKit(parameters.hashIdentifier, key.platformKey(), signature, data);
#else
    return verifyECDSA(parameters.hashIdentifier, key.platformKey(), key.keySizeInBytes(), signature, data);
#endif
}

} // namespace WebCore
