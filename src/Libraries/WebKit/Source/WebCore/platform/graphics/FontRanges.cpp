/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#include "FontRanges.h"

#include "Font.h"
#include "FontSelector.h"
#include <wtf/Assertions.h>
#include <wtf/text/CharacterProperties.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

const Font* FontRanges::Range::font(ExternalResourceDownloadPolicy policy) const
{
    return m_fontAccessor->font(policy);
}

FontRanges::FontRanges(FontRanges&& other, IsGenericFontFamily isGenericFontFamily)
    : m_ranges { WTFMove(other.m_ranges) }
    , m_isGenericFontFamily { isGenericFontFamily }
{
}

class TrivialFontAccessor final : public FontAccessor {
public:
    static Ref<TrivialFontAccessor> create(Ref<Font>&& font)
    {
        return adoptRef(*new TrivialFontAccessor(WTFMove(font)));
    }

private:
    TrivialFontAccessor(RefPtr<Font>&& font)
        : m_font(WTFMove(font))
    {
    }

    const Font* font(ExternalResourceDownloadPolicy) const final
    {
        return m_font.get();
    }

    bool isLoading() const final
    {
        return m_font->isInterstitial();
    }

    RefPtr<Font> m_font;
};

FontRanges::FontRanges(RefPtr<Font>&& font)
{
    if (font)
        m_ranges.append(Range { 0, 0x7FFFFFFF, TrivialFontAccessor::create(font.releaseNonNull()) });
}

FontRanges::~FontRanges() = default;

GlyphData FontRanges::glyphDataForCharacter(char32_t character, ExternalResourceDownloadPolicy policy) const
{
    const Font* resultFont = nullptr;
    if (isGenericFontFamily() && isPrivateUseAreaCharacter(character))
        return GlyphData();

    for (auto& range : m_ranges) {
        if (range.from() <= character && character <= range.to()) {
            if (auto* font = range.font(policy)) {
                if (font->isInterstitial()) {
                    policy = ExternalResourceDownloadPolicy::Forbid;
                    if (!resultFont)
                        resultFont = font;
                } else {
                    auto glyphData = font->glyphDataForCharacter(character);
                    if (glyphData.isValid()) {
                        auto* glyphDataFont = glyphData.font.get();
                        if (glyphDataFont && glyphDataFont->visibility() == Font::Visibility::Visible && resultFont && resultFont->visibility() == Font::Visibility::Invisible)
                            return GlyphData(glyphData.glyph, &glyphDataFont->invisibleFont());
                        return glyphData;
                    }
                }
            }
        }
    }
    if (resultFont) {
        // We want higher-level code to be able to differentiate between
        // "The interstitial font doesn't have the character" and
        // "The real downloaded font doesn't have the character".
        GlyphData result = resultFont->glyphDataForCharacter(character);
        if (!result.font)
            result.font = resultFont;
        return result;
    }
    return GlyphData();
}

const Font* FontRanges::fontForCharacter(char32_t character) const
{
    return glyphDataForCharacter(character, ExternalResourceDownloadPolicy::Allow).font.get();
}

const Font& FontRanges::fontForFirstRange() const
{
    auto* font = m_ranges[0].font(ExternalResourceDownloadPolicy::Forbid);
    ASSERT(font);
    return *font;
}

bool FontRanges::isLoading() const
{
    for (auto& range : m_ranges) {
        if (range.fontAccessor().isLoading())
            return true;
    }
    return false;
}

} // namespace WebCore
