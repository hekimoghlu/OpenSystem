/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

#include "SFrameUtils.h"
#include <wtf/Algorithms.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/ParsingUtilities.h>

namespace WebCore {

#if ASSERT_ENABLED
static constexpr unsigned AES_CM_128_HMAC_SHA256_NONCE_SIZE = 12;
#endif

static inline void writeUInt64(std::span<uint8_t> data, uint64_t value, uint8_t valueLength)
{
    for (unsigned i = 0; i < valueLength; ++i)
        data[i] = (value >> ((valueLength - 1 - i) * 8)) & 0xff;
}

static inline uint64_t readUInt64(std::span<const uint8_t> data)
{
    uint64_t value = 0;
    for (auto byte : data)
        value = (value << 8) | byte;
    return value;
}

static inline uint8_t lengthOfUInt64(uint64_t value)
{
    uint8_t length = 0;
    do {
        ++length;
        value = value >> 8;
    } while (value);
    return length;
}

static inline uint8_t computeFirstHeaderByte(uint64_t keyId, uint64_t counter)
{
    uint8_t value = 0;
    value |= (lengthOfUInt64(counter) - 1) << 4;
    if (keyId < 8)
        value |= keyId;
    else {
        value |= (lengthOfUInt64(keyId) - 1);
        value |= 1 << 3;
    }
    return value;
}

static inline Vector<uint8_t> computeIV(uint64_t counter, const Vector<uint8_t>& saltKey)
{
    // The saltKey is 12 bytes (AES_CM_128_HMAC_SHA256_NONCE_SIZE), we XOR with the counter as 12 bytes.
    // We then extend it to 16 bytes since that is what is expected by the crypto routines.
    Vector<uint8_t> iv(16);
    for (unsigned i = 0; i < 4; ++i)
        iv[i] = saltKey[i];
    for (unsigned i = 11; i >= 4; --i) {
        auto value = counter & 0xff;
        counter = counter >> 8;
        iv[i] = value ^ saltKey[i];
    }
    for (unsigned i = 12; i < 16; ++i)
        iv[i] = 0;
    return iv;
}

static inline bool hasSignature(uint8_t firstByte)
{
    return firstByte & 0x80;
}

static inline bool hasLongKeyLength(uint8_t firstByte)
{
    return firstByte & 0x08;
}

struct SFrameHeaderInfo {
    uint8_t size;
    uint64_t keyId;
    uint64_t counter;
};
static inline std::optional<SFrameHeaderInfo> parseSFrameHeader(std::span<const uint8_t> data)
{
    auto* start = data.data();

    uint64_t keyId = 0;
    uint64_t counter = 0;

    auto firstByte = consume(data);

    // Signature bit.
    if (hasSignature(firstByte))
        return { };

    size_t counterLength = ((firstByte >> 4) & 0x07) + 1;

    if (data.size() < counterLength + 1)
        return { };

    if (hasLongKeyLength(firstByte)) {
        size_t keyLength = (firstByte & 0x07) + 1;
        if (data.size() < counterLength + keyLength + 1)
            return { };

        keyId = readUInt64(consumeSpan(data, keyLength));
        counter = readUInt64(consumeSpan(data, counterLength));
    } else {
        keyId = firstByte & 0x07;
        counter = readUInt64(consumeSpan(data, counterLength));
    }
    uint8_t headerSize = data.data() - start;
    return SFrameHeaderInfo { headerSize, keyId, counter };
}

Ref<RTCRtpSFrameTransformer> RTCRtpSFrameTransformer::create(CompatibilityMode mode)
{
    return adoptRef(*new RTCRtpSFrameTransformer(mode));
}

RTCRtpSFrameTransformer::RTCRtpSFrameTransformer(CompatibilityMode mode)
    : m_compatibilityMode(mode)
{
}

RTCRtpSFrameTransformer::~RTCRtpSFrameTransformer() = default;

ExceptionOr<void> RTCRtpSFrameTransformer::setEncryptionKey(const Vector<uint8_t>& rawKey, std::optional<uint64_t> keyId)
{
    Locker locker { m_keyLock };
    return updateEncryptionKey(rawKey, keyId, ShouldUpdateKeys::Yes);
}

bool RTCRtpSFrameTransformer::hasKey(uint64_t keyId) const
{
    Locker locker { m_keyLock };
    return WTF::anyOf(m_keys, [keyId](auto& key) { return keyId == key.keyId; });
}

ExceptionOr<void> RTCRtpSFrameTransformer::updateEncryptionKey(const Vector<uint8_t>& rawKey, std::optional<uint64_t> keyId, ShouldUpdateKeys shouldUpdateKeys)
{
    ASSERT(m_keyLock.isLocked());

    auto saltKeyResult = computeSaltKey(rawKey);
    if (saltKeyResult.hasException())
        return saltKeyResult.releaseException();

    ASSERT(saltKeyResult.returnValue().size() >= AES_CM_128_HMAC_SHA256_NONCE_SIZE);

    auto authenticationKeyResult = computeAuthenticationKey(rawKey);
    if (authenticationKeyResult.hasException())
        return authenticationKeyResult.releaseException();

    auto encryptionKeyResult = computeEncryptionKey(rawKey);
    if (encryptionKeyResult.hasException())
        return encryptionKeyResult.releaseException();

    if (shouldUpdateKeys == ShouldUpdateKeys::No)
        m_keyId = *keyId;
    else {
        // FIXME: In case keyId is not set, it might be best to use the first non used ID.
        if (!keyId)
            keyId = m_keys.size();

        m_keyId = *keyId;
        m_keys.append({ m_keyId, rawKey });
    }

    m_saltKey = saltKeyResult.releaseReturnValue();
    m_authenticationKey = authenticationKeyResult.releaseReturnValue();
    m_encryptionKey = encryptionKeyResult.releaseReturnValue();

    updateAuthenticationSize();
    m_hasKey = true;

    return { };
}

RTCRtpSFrameTransformer::TransformResult RTCRtpSFrameTransformer::decryptFrame(std::span<const uint8_t> data)
{
    Vector<uint8_t> buffer;
    switch (m_compatibilityMode) {
    case CompatibilityMode::H264: {
        auto offset = computeH264PrefixOffset(data);
        skip(data, offset);
        if (needsRbspUnescaping(data)) {
            buffer = fromRbsp(data);
            data = buffer.span();
        }
        break;
    }
    case CompatibilityMode::VP8: {
        auto offset = computeVP8PrefixOffset(data);
        skip(data, offset);
        break;
    }
    case CompatibilityMode::None:
        break;
    }

    Locker locker { m_keyLock };

    auto header = parseSFrameHeader(data);

    if (!header)
        return makeUnexpected(ErrorInformation {Error::Syntax, "Invalid header"_s, 0 });

    if (header->counter <= m_counter && m_counter)
        return makeUnexpected(ErrorInformation {Error::Syntax, "Invalid counter"_s, 0 });
    m_counter = header->counter;

    if (header->keyId != m_keyId) {
        auto position = m_keys.findIf([keyId = header->keyId](auto& item) { return item.keyId == keyId; });
        if (position == notFound)
            return makeUnexpected(ErrorInformation { Error::KeyID,  "Key ID is unknown"_s, header->keyId });
        auto result = updateEncryptionKey(m_keys[position].keyData, header->keyId, ShouldUpdateKeys::No);
        if (result.hasException())
            return makeUnexpected(ErrorInformation {Error::Other, result.exception().message(), 0 });
    }

    if (data.size() < (header->size + m_authenticationSize))
        return makeUnexpected(ErrorInformation { Error::Syntax, "Chunk is too small for authentication size"_s, 0 });

    auto iv = computeIV(m_counter, m_saltKey);

    // Compute signature
    auto transmittedSignature = data.last(m_authenticationSize);
    auto signature = computeEncryptedDataSignature(iv, data.first(header->size), data.subspan(header->size, data.size() - m_authenticationSize - header->size), m_authenticationKey);
    for (size_t cptr = 0; cptr < m_authenticationSize; ++cptr) {
        if (signature[cptr] != transmittedSignature[cptr]) {
            // FIXME: We should try ratcheting.
            return makeUnexpected(ErrorInformation { Error::Authentication, "Authentication failed"_s, 0 });
        }
    }

    // Decrypt data
    auto dataSize = data.size() - header->size - m_authenticationSize;
    auto result = decryptData(data.subspan(header->size, dataSize), iv, m_encryptionKey);

    if (result.hasException())
        return makeUnexpected(ErrorInformation { Error::Other, result.exception().message(), 0 });

    return result.releaseReturnValue();
}

RTCRtpSFrameTransformer::TransformResult RTCRtpSFrameTransformer::encryptFrame(std::span<const uint8_t> data)
{
    static const unsigned MaxHeaderSize = 17;

    SFrameCompatibilityPrefixBuffer prefixBuffer;
    switch (m_compatibilityMode) {
    case CompatibilityMode::H264:
        prefixBuffer = computeH264PrefixBuffer(data);
        break;
    case CompatibilityMode::VP8:
        prefixBuffer = computeVP8PrefixBuffer(data);
        break;
    case CompatibilityMode::None:
        break;
    }

    auto prefixBufferSpan = WTF::switchOn(prefixBuffer,
        [](const std::span<const uint8_t>& span) -> std::span<const uint8_t> {
            return span;
        },
        [](const Vector<uint8_t>& buffer) -> std::span<const uint8_t> {
            return buffer.span();
        }
    );

    Locker locker { m_keyLock };

    auto iv = computeIV(m_counter, m_saltKey);

    Vector<uint8_t> transformedData(prefixBufferSpan.size() + data.size() + MaxHeaderSize + m_authenticationSize);

    if (prefixBufferSpan.size())
        memcpySpan(transformedData.mutableSpan(), prefixBufferSpan);

    auto newDataSpan = transformedData.mutableSpan().subspan(prefixBufferSpan.size());
    // Fill header.
    size_t headerSize = 1;
    newDataSpan[0] = computeFirstHeaderByte(m_keyId, m_counter);
    if (m_keyId >= 8) {
        auto keyIdLength = lengthOfUInt64(m_keyId);
        writeUInt64(newDataSpan.subspan(headerSize), m_keyId, keyIdLength);
        headerSize += keyIdLength;
    }
    auto counterLength = lengthOfUInt64(m_counter);
    writeUInt64(newDataSpan.subspan(headerSize), m_counter, counterLength);
    headerSize += counterLength;

    ASSERT(headerSize < MaxHeaderSize);
    transformedData.shrink(transformedData.size() - (MaxHeaderSize - headerSize));

    // Fill encrypted data
    auto encryptedData = encryptData(data, iv, m_encryptionKey);
    ASSERT(!encryptedData.hasException());
    if (encryptedData.hasException())
        return makeUnexpected(ErrorInformation { Error::Other, encryptedData.exception().message(), 0 });

    memcpySpan(newDataSpan.subspan(headerSize), encryptedData.returnValue().span().first(data.size()));

    // Fill signature
    auto signature = computeEncryptedDataSignature(iv, newDataSpan.first(headerSize), newDataSpan.subspan(headerSize, data.size()), m_authenticationKey);
    memcpySpan(newDataSpan.subspan(data.size() + headerSize), signature.span().first(m_authenticationSize));

    if (m_compatibilityMode == CompatibilityMode::H264)
        toRbsp(transformedData, prefixBufferSpan.size());

    ++m_counter;

    return transformedData;
}

RTCRtpSFrameTransformer::TransformResult RTCRtpSFrameTransformer::transform(std::span<const uint8_t> data)
{
    if (!m_hasKey)
        return makeUnexpected(ErrorInformation { Error::KeyID,  "Key is not initialized"_s, 0 });

    return m_isEncrypting ? encryptFrame(data) : decryptFrame(data);
}

#if !PLATFORM(COCOA) && !USE(GSTREAMER_WEBRTC)
ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::computeSaltKey(const Vector<uint8_t>&)
{
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::computeAuthenticationKey(const Vector<uint8_t>&)
{
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::computeEncryptionKey(const Vector<uint8_t>&)
{
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::decryptData(std::span<const uint8_t>, const Vector<uint8_t>&, const Vector<uint8_t>&)
{
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<Vector<uint8_t>> RTCRtpSFrameTransformer::encryptData(std::span<const uint8_t>, const Vector<uint8_t>&, const Vector<uint8_t>&)
{
    return Exception { ExceptionCode::NotSupportedError };
}

Vector<uint8_t> RTCRtpSFrameTransformer::computeEncryptedDataSignature(const Vector<uint8_t>&, std::span<const uint8_t>, std::span<const uint8_t>, const Vector<uint8_t>&)
{
    return { };
}

void RTCRtpSFrameTransformer::updateAuthenticationSize()
{
}
#endif // !PLATFORM(COCOA) && !USE(GSTREAMER_WEBRTC)

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
