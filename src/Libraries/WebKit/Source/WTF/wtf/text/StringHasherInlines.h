/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
#pragma once

#include <wtf/Assertions.h>
#include <wtf/text/StringHasher.h>
#include <wtf/text/SuperFastHash.h>
#include <wtf/text/WYHash.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

template<typename T, typename Converter>
unsigned StringHasher::computeHashAndMaskTop8Bits(std::span<const T> data)
{
#if ENABLE(WYHASH_STRING_HASHER)
    if (LIKELY(data.size() <= smallStringThreshold))
        return SuperFastHash::computeHashAndMaskTop8Bits<T, Converter>(data);
    return WYHash::computeHashAndMaskTop8Bits<T, Converter>(data);
#else
    return SuperFastHash::computeHashAndMaskTop8Bits<T, Converter>(data);
#endif
}

template<typename T, unsigned characterCount>
constexpr unsigned StringHasher::computeLiteralHashAndMaskTop8Bits(const T (&characters)[characterCount])
{
    constexpr unsigned characterCountWithoutNull = characterCount - 1;
#if ENABLE(WYHASH_STRING_HASHER)
    if constexpr (LIKELY(characterCountWithoutNull <= smallStringThreshold))
        return SuperFastHash::computeHashAndMaskTop8Bits<T>(std::span { characters, characterCountWithoutNull });
    return WYHash::computeHashAndMaskTop8Bits<T>(std::span { characters, characterCountWithoutNull });
#else
    return SuperFastHash::computeHashAndMaskTop8Bits<T>(std::span { characters, characterCountWithoutNull });
#endif
}

inline void StringHasher::addCharacter(UChar character)
{
#if ENABLE(WYHASH_STRING_HASHER)
    if (m_bufferSize == smallStringThreshold) {
        // This algorithm must stay in sync with WYHash::hash function.
        if (!m_pendingHashValue) {
            m_seed = WYHash::initSeed();
            m_see1 = m_seed;
            m_see2 = m_seed;
            m_pendingHashValue = true;
        }
        UChar* p = m_buffer.data();
        while (m_bufferSize >= 24) {
            WYHash::consume24Characters(p, WYHash::Reader16Bit<UChar>::wyr8, m_seed, m_see1, m_see2);
            p += 24;
            m_bufferSize -= 24;
        }
        ASSERT(!m_bufferSize);
        m_numberOfProcessedCharacters += smallStringThreshold;
    }

    ASSERT(m_bufferSize < smallStringThreshold);
    m_buffer[m_bufferSize++] = character;
#else
    SuperFastHash::addCharacterImpl(character, m_hasPendingCharacter, m_pendingCharacter, m_hash);
#endif
}

inline unsigned StringHasher::hashWithTop8BitsMasked()
{
#if ENABLE(WYHASH_STRING_HASHER)
    unsigned hashValue;
    if (!m_pendingHashValue) {
        ASSERT(m_bufferSize <= smallStringThreshold);
        hashValue = SuperFastHash::computeHashAndMaskTop8Bits<UChar>(std::span { m_buffer }.first(m_bufferSize));
    } else {
        // This algorithm must stay in sync with WYHash::hash function.
        auto wyr8 = WYHash::Reader16Bit<UChar>::wyr8;
        unsigned i = m_bufferSize;
        if (i <= 24)
            m_seed ^= m_see1 ^ m_see2;
        UChar* p = m_buffer.data();
        WYHash::handleGreaterThan8CharactersCase(p, i, wyr8, m_seed, m_see1, m_see2);

        uint64_t a = 0;
        uint64_t b = 0;
        if (m_bufferSize >= 8) {
            a = wyr8(p + i - 8);
            b = wyr8(p + i - 4);
        } else {
            UChar tmp[8];
            unsigned bufferIndex = smallStringThreshold - (8 - i);
            for (unsigned tmpIndex = 0; tmpIndex < 8; tmpIndex++) {
                tmp[tmpIndex] = m_buffer[bufferIndex];
                bufferIndex = (bufferIndex + 1) % smallStringThreshold;
            }

            UChar* tmpPtr = tmp;
            a = wyr8(tmpPtr);
            b = wyr8(tmpPtr + 4);
        }

        const uint64_t totalByteCount = (static_cast<uint64_t>(m_numberOfProcessedCharacters) + static_cast<uint64_t>(m_bufferSize)) << 1;
        hashValue = StringHasher::avoidZero(WYHash::handleEndCase(a, b, m_seed, totalByteCount) & StringHasher::maskHash);

        m_pendingHashValue = false;
        m_numberOfProcessedCharacters = m_seed = m_see1 = m_see2 = 0;
    }
    m_bufferSize = 0;
    return hashValue;
#else
    unsigned hashValue = SuperFastHash::hashWithTop8BitsMaskedImpl(m_hasPendingCharacter, m_pendingCharacter, m_hash);
    m_hasPendingCharacter = false;
    m_pendingCharacter = 0;
    m_hash = stringHashingStartValue;
    return hashValue;
#endif
}

} // namespace WTF

using WTF::StringHasher;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
