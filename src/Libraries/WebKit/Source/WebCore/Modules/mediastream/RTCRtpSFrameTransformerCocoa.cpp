/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
#include "RTCRtpSFrameTransformer.h"

#if ENABLE(WEB_RTC)

#include "CryptoUtilitiesCocoa.h"
#include "SFrameUtils.h"
#include <CommonCrypto/CommonCrypto.h>

namespace WebCore {

ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::computeSaltKey(const Vector<uint8_t>& rawKey)
{
    return deriveHDKFSHA256Bits(rawKey.subspan(0, 16), "SFrame10"_span8, "salt"_span8, 96);
}

static ExceptionOr<Vector<uint8_t>> createBaseSFrameKey(const Vector<uint8_t>& rawKey)
{
    return deriveHDKFSHA256Bits(rawKey.subspan(0, 16), "SFrame10"_span8, "key"_span8, 128);
}

ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::computeAuthenticationKey(const Vector<uint8_t>& rawKey)
{
    auto key = createBaseSFrameKey(rawKey);
    if (key.hasException())
        return key;

    return deriveHDKFSHA256Bits(key.returnValue().subspan(0, 16), "SFrame10 AES CM AEAD"_span8, "auth"_span8, 256);
}

ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::computeEncryptionKey(const Vector<uint8_t>& rawKey)
{
    auto key = createBaseSFrameKey(rawKey);
    if (key.hasException())
        return key;

    return deriveHDKFSHA256Bits(key.returnValue().subspan(0, 16), "SFrame10 AES CM AEAD"_span8, "enc"_span8, 128);
}

ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::decryptData(std::span<const uint8_t> data, const Vector<uint8_t>& iv, const Vector<uint8_t>& key)
{
    return transformAESCTR(kCCDecrypt, iv, iv.size(), key, data);
}

ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::encryptData(std::span<const uint8_t> data, const Vector<uint8_t>& iv, const Vector<uint8_t>& key)
{
    return transformAESCTR(kCCEncrypt, iv, iv.size(), key, data);
}

Vector<uint8_t> RTCRtpSFrameTransformer::computeEncryptedDataSignature(const Vector<uint8_t>& nonce, std::span<const uint8_t> header, std::span<const uint8_t> data, const Vector<uint8_t>& key)
{
    auto headerLength = encodeBigEndian(header.size());
    auto dataLength = encodeBigEndian(data.size());

    Vector<uint8_t> result(CC_SHA256_DIGEST_LENGTH);
    CCHmacContext context;
    CCHmacInit(&context, kCCHmacAlgSHA256, key.data(), key.size());
    CCHmacUpdate(&context, headerLength.data(), headerLength.size());
    CCHmacUpdate(&context, dataLength.data(), dataLength.size());
    CCHmacUpdate(&context, nonce.data(), 12);
    CCHmacUpdate(&context, header.data(), header.size());
    CCHmacUpdate(&context, data.data(), data.size());
    CCHmacFinal(&context, result.data());

    return result;
}

void RTCRtpSFrameTransformer::updateAuthenticationSize()
{
    if (m_authenticationSize > CC_SHA256_DIGEST_LENGTH)
        m_authenticationSize = CC_SHA256_DIGEST_LENGTH;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
