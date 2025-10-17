/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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

#include <span>
#include <wtf/Compiler.h>
#include <wtf/text/StringHasher.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

// Paul Hsieh's SuperFastHash
// http://www.azillionmonkeys.com/qed/hash.html

// LChar data is interpreted as Latin-1-encoded (zero-extended to 16 bits).

// NOTE: The hash computation here must stay in sync with the create_hash_table script in
// JavaScriptCore and the CodeGeneratorJS.pm script in WebCore.

class SuperFastHash {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static constexpr unsigned flagCount = StringHasher::flagCount;
    static constexpr unsigned maskHash = StringHasher::maskHash;
    typedef StringHasher::DefaultConverter DefaultConverter;

    SuperFastHash() = default;

    // The hasher hashes two characters at a time, and thus an "aligned" hasher is one
    // where an even number of characters have been added. Callers that always add
    // characters two at a time can use the "assuming aligned" functions.
    void addCharactersAssumingAligned(UChar a, UChar b)
    {
        ASSERT(!m_hasPendingCharacter);
        addCharactersAssumingAlignedImpl(a, b, m_hash);
    }

    void addCharacter(UChar character)
    {
        addCharacterImpl(character, m_hasPendingCharacter, m_pendingCharacter, m_hash);
    }

    void addCharacters(UChar a, UChar b)
    {
        if (m_hasPendingCharacter) {
#if ASSERT_ENABLED
            m_hasPendingCharacter = false;
#endif
            addCharactersAssumingAligned(m_pendingCharacter, a);
            m_pendingCharacter = b;
#if ASSERT_ENABLED
            m_hasPendingCharacter = true;
#endif
            return;
        }

        addCharactersAssumingAligned(a, b);
    }

    template<typename T, typename Converter = DefaultConverter>
    void addCharactersAssumingAligned(const T* data, unsigned length)
    {
        ASSERT(!m_hasPendingCharacter);

        bool remainder = length & 1;
        length >>= 1;

        while (length--) {
            addCharactersAssumingAligned(Converter::convert(data[0]), Converter::convert(data[1]));
            data += 2;
        }

        if (remainder)
            addCharacter(Converter::convert(*data));
    }

    template<typename T, typename Converter = DefaultConverter>
    void addCharactersAssumingAligned(const T* data)
    {
        ASSERT(!m_hasPendingCharacter);

        while (T a = *data++) {
            T b = *data++;
            if (!b) {
                addCharacter(Converter::convert(a));
                break;
            }
            addCharactersAssumingAligned(Converter::convert(a), Converter::convert(b));
        }
    }

    template<typename T, typename Converter = DefaultConverter>
    void addCharacters(const T* data, unsigned length)
    {
        if (!length)
            return;
        if (m_hasPendingCharacter) {
            m_hasPendingCharacter = false;
            addCharactersAssumingAligned(m_pendingCharacter, Converter::convert(*data++));
            --length;
        }
        addCharactersAssumingAligned<T, Converter>(data, length);
    }

    template<typename T, typename Converter = DefaultConverter>
    void addCharacters(std::span<const T> data)
    {
        addCharacters(data.data(), data.size());
    }

    template<typename T, typename Converter = DefaultConverter>
    void addCharacters(const T* data)
    {
        if (m_hasPendingCharacter && *data) {
            m_hasPendingCharacter = false;
            addCharactersAssumingAligned(m_pendingCharacter, Converter::convert(*data++));
        }
        addCharactersAssumingAligned<T, Converter>(data);
    }

    unsigned hashWithTop8BitsMasked() const
    {
        return hashWithTop8BitsMaskedImpl(m_hasPendingCharacter, m_pendingCharacter, m_hash);
    }

    unsigned hash() const
    {
        return StringHasher::finalize(processPendingCharacter());
    }

    template<typename T, typename Converter = DefaultConverter>
    ALWAYS_INLINE static constexpr unsigned computeHashAndMaskTop8Bits(std::span<const T> data)
    {
        return StringHasher::finalizeAndMaskTop8Bits(computeHashImpl<T, Converter>(data));
    }

    template<typename T, typename Converter = DefaultConverter>
    ALWAYS_INLINE static constexpr unsigned computeHashAndMaskTop8Bits(const T* data)
    {
        return StringHasher::finalizeAndMaskTop8Bits(computeHashImpl<T, Converter>(data));
    }

    template<typename T, typename Converter = DefaultConverter>
    static constexpr unsigned computeHash(std::span<const T> data)
    {
        return StringHasher::finalize(computeHashImpl<T, Converter>(data));
    }

    template<typename T, typename Converter = DefaultConverter>
    static constexpr unsigned computeHash(const T* data)
    {
        return StringHasher::finalize(computeHashImpl<T, Converter>(data));
    }

    template<typename T, unsigned charactersCount>
    static constexpr unsigned computeLiteralHash(const T (&characters)[charactersCount])
    {
        return computeHash<T>(characters, charactersCount - 1);
    }

    template<typename T, unsigned charactersCount>
    static constexpr unsigned computeLiteralHashAndMaskTop8Bits(const T (&characters)[charactersCount])
    {
        return computeHashAndMaskTop8Bits<T>(characters, charactersCount - 1);
    }

private:
    friend class StringHasher;

    static void addCharactersAssumingAlignedImpl(UChar a, UChar b, unsigned& hash)
    {
        hash = calculateWithTwoCharacters(hash, a, b);
    }

    static void addCharacterImpl(UChar character, bool& hasPendingCharacter, UChar& pendingCharacter, unsigned& hash)
    {
        if (hasPendingCharacter) {
            hasPendingCharacter = false;
            addCharactersAssumingAlignedImpl(pendingCharacter, character, hash);
            return;
        }

        pendingCharacter = character;
        hasPendingCharacter = true;
    }

    static constexpr unsigned calculateWithRemainingLastCharacter(unsigned hash, unsigned character)
    {
        unsigned result = hash;

        result += character;
        result ^= result << 11;
        result += result >> 17;

        return result;
    }

    ALWAYS_INLINE static constexpr unsigned calculateWithTwoCharacters(unsigned hash, unsigned firstCharacter, unsigned secondCharacter)
    {
        unsigned result = hash;

        result += firstCharacter;
        result = (result << 16) ^ ((secondCharacter << 11) ^ result);
        result += result >> 11;

        return result;
    }

    template<typename T, typename Converter>
    static constexpr unsigned computeHashImpl(std::span<const T> characters)
    {
        unsigned result = stringHashingStartValue;
        for (size_t i = 0; i + 1 < characters.size(); i += 2)
            result = calculateWithTwoCharacters(result, Converter::convert(characters[i]), Converter::convert(characters[i + 1]));
        if (characters.size() % 2)
            return calculateWithRemainingLastCharacter(result, Converter::convert(characters.back()));
        return result;
    }

    template<typename T, typename Converter>
    static constexpr unsigned computeHashImpl(const T* characters)
    {
        unsigned result = stringHashingStartValue;
        while (T a = *characters++) {
            T b = *characters++;
            if (!b)
                return calculateWithRemainingLastCharacter(result, Converter::convert(a));
            result = calculateWithTwoCharacters(result, Converter::convert(a), Converter::convert(b));
        }
        return result;
    }

    unsigned processPendingCharacter() const
    {
        return processPendingCharacterImpl(m_hasPendingCharacter, m_pendingCharacter, m_hash);
    }

    static unsigned hashWithTop8BitsMaskedImpl(const bool hasPendingCharacter, const UChar pendingCharacter, const unsigned hash)
    {
        return StringHasher::finalizeAndMaskTop8Bits(processPendingCharacterImpl(hasPendingCharacter, pendingCharacter, hash));
    }

    static unsigned processPendingCharacterImpl(const bool hasPendingCharacter, const UChar pendingCharacter, const unsigned hash)
    {
        unsigned result = hash;

        // Handle end case.
        if (hasPendingCharacter)
            return calculateWithRemainingLastCharacter(result, pendingCharacter);
        return result;
    }

    unsigned m_hash { stringHashingStartValue };
    UChar m_pendingCharacter { 0 };
    bool m_hasPendingCharacter { false };
};

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

using WTF::SuperFastHash;
