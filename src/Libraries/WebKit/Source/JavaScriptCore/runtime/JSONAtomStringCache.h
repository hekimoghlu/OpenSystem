/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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

#include <wtf/text/AtomStringImpl.h>

namespace JSC {

class VM;

class JSONAtomStringCache {
public:
    static constexpr auto maxStringLengthForCache = 27;
    static constexpr auto capacity = 256;

    struct Slot {
        UChar m_buffer[maxStringLengthForCache] { };
        UChar m_length { 0 };
        RefPtr<AtomStringImpl> m_impl;
    };
    static_assert(sizeof(Slot) <= 64);

    using Cache = std::array<Slot, capacity>;

    template<typename CharacterType>
    ALWAYS_INLINE Ref<AtomStringImpl> makeIdentifier(std::span<const CharacterType> characters)
    {
        return make(characters);
    }

    ALWAYS_INLINE void clear()
    {
        m_cache.fill({ });
    }

    VM& vm() const;

private:
    template<typename CharacterType>
    Ref<AtomStringImpl> make(std::span<const CharacterType>);

    ALWAYS_INLINE Slot& cacheSlot(UChar firstCharacter, UChar lastCharacter, UChar length)
    {
        unsigned hash = (firstCharacter << 6) ^ ((lastCharacter << 14) ^ firstCharacter);
        hash += (hash >> 14) + (length << 14);
        hash ^= hash << 14;
        return m_cache[(hash + (hash >> 6)) % capacity];
    }

    Cache m_cache { };
};

} // namespace JSC
