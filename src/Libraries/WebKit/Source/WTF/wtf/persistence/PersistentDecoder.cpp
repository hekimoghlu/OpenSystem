/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#include <wtf/persistence/PersistentDecoder.h>

#include <wtf/StdLibExtras.h>
#include <wtf/persistence/PersistentEncoder.h>

namespace WTF::Persistence {

Decoder::Decoder(std::span<const uint8_t> span)
    : m_buffer(span)
    , m_bufferPosition(span.begin())
{
}

Decoder::~Decoder() = default;

bool Decoder::bufferIsLargeEnoughToContain(size_t size) const
{
    return size <= static_cast<size_t>(std::distance(m_bufferPosition, m_buffer.end()));
}

std::span<const uint8_t> Decoder::bufferPointerForDirectRead(size_t size)
{
    if (!bufferIsLargeEnoughToContain(size))
        return { };

    auto data = m_buffer.subspan(currentOffset(), size);
    m_bufferPosition += size;

    Encoder::updateChecksumForData(m_sha1, data);
    return data;
}

bool Decoder::decodeFixedLengthData(std::span<uint8_t> span)
{
    auto buffer = bufferPointerForDirectRead(span.size());
    if (!buffer.data())
        return false;
    memcpySpan(span, buffer);
    return true;
}

bool Decoder::rewind(size_t size)
{
    if (size <= currentOffset()) {
        m_bufferPosition -= size;
        return true;
    }
    return false;
}

template<typename T>
Decoder& Decoder::decodeNumber(std::optional<T>& optional)
{
    if (!bufferIsLargeEnoughToContain(sizeof(T)))
        return *this;

    T value;
    memcpySpan(asMutableByteSpan(value), m_buffer.subspan(currentOffset(), sizeof(T)));
    m_bufferPosition += sizeof(T);

    Encoder::updateChecksumForNumber(m_sha1, value);
    optional = value;
    return *this;
}

Decoder& Decoder::operator>>(std::optional<bool>& result)
{
    return decodeNumber(result);
}

Decoder& Decoder::operator>>(std::optional<uint8_t>& result)
{
    return decodeNumber(result);
}

Decoder& Decoder::operator>>(std::optional<uint16_t>& result)
{
    return decodeNumber(result);
}

Decoder& Decoder::operator>>(std::optional<int16_t>& result)
{
    return decodeNumber(result);
}

Decoder& Decoder::operator>>(std::optional<uint32_t>& result)
{
    return decodeNumber(result);
}

Decoder& Decoder::operator>>(std::optional<uint64_t>& result)
{
    return decodeNumber(result);
}

Decoder& Decoder::operator>>(std::optional<int32_t>& result)
{
    return decodeNumber(result);
}

Decoder& Decoder::operator>>(std::optional<int64_t>& result)
{
    return decodeNumber(result);
}

Decoder& Decoder::operator>>(std::optional<float>& result)
{
    return decodeNumber(result);
}

Decoder& Decoder::operator>>(std::optional<double>& result)
{
    return decodeNumber(result);
}

bool Decoder::verifyChecksum()
{
    SHA1::Digest computedHash;
    m_sha1.computeHash(computedHash);

    SHA1::Digest savedHash;
    if (!decodeFixedLengthData({ savedHash }))
        return false;

    return computedHash == savedHash;
}

} // namespace WTF::Persistence
