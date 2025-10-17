/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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

#include "Color.h"
#include "Gradient.h"
#include <variant>
#include <wtf/HashFunctions.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

struct FontPaletteIndex {
    enum class Type : uint8_t;

    FontPaletteIndex() = default;

    FontPaletteIndex(Type type)
        : type(type)
    {
        ASSERT(type == Type::Light || type == Type::Dark);
    }

    FontPaletteIndex(unsigned integer)
        : type(Type::Integer)
        , integer(integer)
    {
    }

    operator bool() const
    {
        return type != Type::Integer || integer;
    }

    bool operator==(const FontPaletteIndex& other) const
    {
        if (type != other.type)
            return false;
        if (type == Type::Integer)
            return integer == other.integer;
        return true;
    }

    enum class Type : uint8_t {
        Light,
        Dark,
        Integer
    };
    Type type { Type::Integer };

    unsigned integer { 0 };
};

inline void add(Hasher& hasher, const FontPaletteIndex& paletteIndex)
{
    add(hasher, paletteIndex.type);
    if (paletteIndex.type == FontPaletteIndex::Type::Integer)
        add(hasher, paletteIndex.integer);
}

class FontPaletteValues {
public:
    using OverriddenColor = std::pair<unsigned, Color>;

    FontPaletteValues() = default;

    FontPaletteValues(std::optional<FontPaletteIndex> basePalette, Vector<OverriddenColor>&& overrideColors)
        : m_basePalette(basePalette)
        , m_overrideColors(WTFMove(overrideColors))
    {
    }

    std::optional<FontPaletteIndex> basePalette() const
    {
        return m_basePalette;
    }

    const Vector<OverriddenColor>& overrideColors() const
    {
        return m_overrideColors;
    }

    operator bool() const
    {
        return m_basePalette || !m_overrideColors.isEmpty();
    }

    bool operator==(const FontPaletteValues& other) const
    {
        return m_basePalette == other.m_basePalette && m_overrideColors == other.m_overrideColors;
    }

private:
    std::optional<FontPaletteIndex> m_basePalette;
    Vector<OverriddenColor> m_overrideColors;
};

inline void add(Hasher& hasher, const FontPaletteValues& fontPaletteValues)
{
    add(hasher, fontPaletteValues.basePalette());
    add(hasher, fontPaletteValues.overrideColors());
}

} // namespace WebCore

namespace WTF {

template<> struct DefaultHash<WebCore::FontPaletteValues> {
    static unsigned hash(const WebCore::FontPaletteValues& key) { return computeHash(key); }
    static bool equal(const WebCore::FontPaletteValues& a, const WebCore::FontPaletteValues& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

}
