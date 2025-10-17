/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
#include <wtf/persistence/PersistentEncoder.h>

#include <wtf/SHA1.h>
#include <wtf/StdLibExtras.h>

namespace WTF::Persistence {

Encoder::Encoder() = default;

Encoder::~Encoder() = default;

std::span<uint8_t> Encoder::grow(size_t size)
{
    size_t newPosition = m_buffer.size();
    m_buffer.grow(m_buffer.size() + size);
    return m_buffer.mutableSpan().subspan(newPosition);
}

void Encoder::updateChecksumForData(SHA1& sha1, std::span<const uint8_t> span)
{
    auto typeSalt = Salt<uint8_t*>::value;
    sha1.addBytes(asByteSpan(typeSalt));
    sha1.addBytes(span);
}

void Encoder::encodeFixedLengthData(std::span<const uint8_t> span)
{
    updateChecksumForData(m_sha1, span);

    memcpySpan(grow(span.size()), span);
}

template<typename Type>
Encoder& Encoder::encodeNumber(Type value)
{
    Encoder::updateChecksumForNumber(m_sha1, value);

    memcpySpan(grow(sizeof(Type)), asByteSpan(value));
    return *this;
}

Encoder& Encoder::operator<<(bool value)
{
    return encodeNumber(value);
}

Encoder& Encoder::operator<<(uint8_t value)
{
    return encodeNumber(value);
}

Encoder& Encoder::operator<<(uint16_t value)
{
    return encodeNumber(value);
}

Encoder& Encoder::operator<<(int16_t value)
{
    return encodeNumber(value);
}

Encoder& Encoder::operator<<(uint32_t value)
{
    return encodeNumber(value);
}

Encoder& Encoder::operator<<(uint64_t value)
{
    return encodeNumber(value);
}

Encoder& Encoder::operator<<(int32_t value)
{
    return encodeNumber(value);
}

Encoder& Encoder::operator<<(int64_t value)
{
    return encodeNumber(value);
}

Encoder& Encoder::operator<<(float value)
{
    return encodeNumber(value);
}

Encoder& Encoder::operator<<(double value)
{
    return encodeNumber(value);
}

void Encoder::encodeChecksum()
{
    SHA1::Digest hash;
    m_sha1.computeHash(hash);
    encodeFixedLengthData(hash);
}

}
