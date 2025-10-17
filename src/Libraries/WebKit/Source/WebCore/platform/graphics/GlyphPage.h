/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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

#include "Font.h"
#include "Glyph.h"
#include "TextFlags.h"
#include <unicode/utypes.h>
#include <wtf/BitSet.h>
#include <wtf/CheckedPtr.h>
#include <wtf/RefCounted.h>
#include <wtf/Ref.h>

namespace WebCore {

// Holds the glyph index and the corresponding Font information for a given
// character.
struct GlyphData {
    GlyphData(Glyph glyph = 0, const Font* font = nullptr, ColorGlyphType colorGlyphType = ColorGlyphType::Outline)
        : glyph(glyph)
        , colorGlyphType(colorGlyphType)
        , font(font)
    {
    }

    bool isValid() const { return !!font; }

    Glyph glyph;
    ColorGlyphType colorGlyphType;
    SingleThreadWeakPtr<const Font> font;
};

// A GlyphPage contains a fixed-size set of GlyphData mappings for a contiguous
// range of characters in the Unicode code space. GlyphPages are indexed
// starting from 0 and incrementing for each "size" number of glyphs.
class GlyphPage final : public RefCounted<GlyphPage> {
public:
    static Ref<GlyphPage> create(const Font& font)
    {
        return adoptRef(*new GlyphPage(font));
    }

    ~GlyphPage()
    {
        --s_count;
    }

    static unsigned count() { return s_count; }

    static constexpr unsigned size = 16;

    static constexpr unsigned sizeForPageNumber(unsigned) { return size; }
    static constexpr unsigned indexForCodePoint(char32_t c) { return c % size; }
    static constexpr unsigned pageNumberForCodePoint(char32_t c) { return c / size; }
    static constexpr char32_t startingCodePointInPageNumber(unsigned pageNumber) { return pageNumber * size; }
    static constexpr bool pageNumberIsUsedForArabic(unsigned pageNumber) { return startingCodePointInPageNumber(pageNumber) >= 0x600 && startingCodePointInPageNumber(pageNumber) + sizeForPageNumber(pageNumber) < 0x700; }

    GlyphData glyphDataForCharacter(char32_t c) const
    {
        return glyphDataForIndex(indexForCodePoint(c));
    }

    Glyph glyphForCharacter(char32_t c) const
    {
        return glyphForIndex(indexForCodePoint(c));
    }

    GlyphData glyphDataForIndex(unsigned index) const
    {
        Glyph glyph = glyphForIndex(index);
        auto colorGlyphType = colorGlyphTypeForIndex(index);
        return GlyphData(glyph, glyph ? m_font.get() : nullptr, colorGlyphType);
    }

    Glyph glyphForIndex(unsigned index) const
    {
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(index < size);
        return m_glyphs[index];
    }

    ColorGlyphType colorGlyphTypeForIndex(unsigned index) const
    {
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(index < size);
        return m_isColor.get(index) ? ColorGlyphType::Color : ColorGlyphType::Outline;
    }

    // FIXME: Pages are immutable after initialization. This should be private.
    void setGlyphForIndex(unsigned index, Glyph glyph, ColorGlyphType colorGlyphType)
    {
        ASSERT_WITH_SECURITY_IMPLICATION(index < size);
        m_glyphs[index] = glyph;
        m_isColor.set(index, colorGlyphType == ColorGlyphType::Color);
    }

    const Font& font() const
    {
        return *m_font;
    }

    // Implemented by the platform.
    bool fill(std::span<const UChar> characterBuffer);

private:
    explicit GlyphPage(const Font& font)
        : m_font(font)
    {
        ++s_count;
    }

    SingleThreadWeakPtr<const Font> m_font;
    std::array<Glyph, size> m_glyphs { };
    WTF::BitSet<size> m_isColor;

    WEBCORE_EXPORT static unsigned s_count;
};

} // namespace WebCore
