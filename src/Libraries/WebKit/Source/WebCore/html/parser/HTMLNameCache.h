/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

#include "QualifiedName.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class HTMLNameCache {
public:
    ALWAYS_INLINE static QualifiedName makeAttributeQualifiedName(std::span<const UChar> string)
    {
        return makeQualifiedName(string);
    }

    ALWAYS_INLINE static QualifiedName makeAttributeQualifiedName(std::span<const LChar> string)
    {
        return makeQualifiedName(string);
    }

    ALWAYS_INLINE static AtomString makeAttributeValue(std::span<const UChar> string)
    {
        return makeAtomString(string);
    }

    ALWAYS_INLINE static AtomString makeAttributeValue(std::span<const LChar> string)
    {
        return makeAtomString(string);
    }

    ALWAYS_INLINE static void clear()
    {
        // FIXME (webkit.org/b/230019): We should try to find more opportunities to clear this cache without hindering this performance optimization.
        atomStringCache().fill({ });
        qualifiedNameCache().fill({ });
    }

private:
    template<typename CharacterType>
    ALWAYS_INLINE static AtomString makeAtomString(std::span<const CharacterType> string)
    {
        if (string.empty())
            return emptyAtom();

        if (string.size() > maxStringLengthForCache)
            return AtomString(string);

        auto& slot = atomStringCacheSlot(string.front(), string.back(), string.size());
        if (!equal(slot.impl(), string)) {
            AtomString result { string };
            slot = result;
            return result;
        }

        return slot;
    }

    template<typename CharacterType>
    ALWAYS_INLINE static QualifiedName makeQualifiedName(std::span<const CharacterType> string)
    {
        if (string.empty())
            return nullQName();

        if (string.size() > maxStringLengthForCache)
            return QualifiedName(nullAtom(), AtomString(string), nullAtom());

        auto& slot = qualifiedNameCacheSlot(string.front(), string.back(), string.size());
        if (!slot || !equal(slot->m_localName.impl(), string)) {
            QualifiedName result(nullAtom(), AtomString(string), nullAtom());
            slot = result.impl();
            return result;
        }

        return *slot;
    }

    ALWAYS_INLINE static size_t slotIndex(UChar firstCharacter, UChar lastCharacter, UChar length)
    {
        unsigned hash = (firstCharacter << 6) ^ ((lastCharacter << 14) ^ firstCharacter);
        hash += (hash >> 14) + (length << 14);
        hash ^= hash << 14;
        return (hash + (hash >> 6)) % capacity;
    }

    ALWAYS_INLINE static AtomString& atomStringCacheSlot(UChar firstCharacter, UChar lastCharacter, UChar length)
    {
        auto index = slotIndex(firstCharacter, lastCharacter, length);
        return atomStringCache()[index];
    }

    ALWAYS_INLINE static RefPtr<QualifiedName::QualifiedNameImpl>& qualifiedNameCacheSlot(UChar firstCharacter, UChar lastCharacter, UChar length)
    {
        auto index = slotIndex(firstCharacter, lastCharacter, length);
        return qualifiedNameCache()[index];
    }

    static constexpr auto maxStringLengthForCache = 36;
    static constexpr auto capacity = 512;

    using AtomStringCache = std::array<AtomString, capacity>;
    using QualifiedNameCache = std::array<RefPtr<QualifiedName::QualifiedNameImpl>, capacity>;

    static AtomStringCache& atomStringCache();
    static QualifiedNameCache& qualifiedNameCache();
};

} // namespace WebCore
