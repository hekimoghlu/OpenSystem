/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#include "FontCascade.h"

#if USE(SKIA)
#include "FontCache.h"
#include "GraphicsContextSkia.h"
#include "SurrogatePairAwareTextIterator.h"
#include <skia/core/SkTextBlob.h>
#include <wtf/text/CharacterProperties.h>

namespace WebCore {

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GLib/Win port

void FontCascade::drawGlyphs(GraphicsContext& graphicsContext, const Font& font, std::span<const GlyphBufferGlyph> glyphs, std::span<const GlyphBufferAdvance> advances, const FloatPoint& position, FontSmoothingMode smoothingMode)
{
    if (!font.platformData().size())
        return;

    const auto& fontPlatformData = font.platformData();
    const auto& skFont = fontPlatformData.skFont();

    if (!font.allowsAntialiasing())
        smoothingMode = FontSmoothingMode::NoSmoothing;

    SkFont::Edging edging;
    switch (smoothingMode) {
    case FontSmoothingMode::AutoSmoothing:
        edging = skFont.getEdging();
        break;
    case FontSmoothingMode::Antialiased:
        edging = SkFont::Edging::kAntiAlias;
        break;
    case FontSmoothingMode::SubpixelAntialiased:
        edging = SkFont::Edging::kSubpixelAntiAlias;
        break;
    case FontSmoothingMode::NoSmoothing:
        edging = SkFont::Edging::kAlias;
        break;
    }

    bool isVertical = fontPlatformData.orientation() == FontOrientation::Vertical;
    SkTextBlobBuilder builder;
    const auto& buffer = [&]() {
        if (skFont.getEdging() == edging)
            return isVertical ? builder.allocRunPos(skFont, glyphs.size()) : builder.allocRunPosH(skFont, glyphs.size(), 0);

        SkFont copiedFont = skFont;
        copiedFont.setEdging(edging);
        return isVertical ? builder.allocRunPos(copiedFont, glyphs.size()) : builder.allocRunPosH(copiedFont, glyphs.size(), 0);
    }();

    FloatSize glyphPosition;
    for (size_t i = 0; i < glyphs.size(); ++i) {
        buffer.glyphs[i] = glyphs[i];

        if (isVertical) {
            glyphPosition += advances[i];
            buffer.pos[2 * i] = glyphPosition.height();
            buffer.pos[2 * i + 1] = glyphPosition.width();
        } else {
            buffer.pos[i] = glyphPosition.width();
            glyphPosition += advances[i];
        }
    }

    auto blob = builder.make();
    static_cast<GraphicsContextSkia*>(&graphicsContext)->drawSkiaText(blob, SkFloatToScalar(position.x()), SkFloatToScalar(position.y()), edging != SkFont::Edging::kAlias, isVertical);
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

bool FontCascade::canReturnFallbackFontsForComplexText()
{
    return false;
}

bool FontCascade::canExpandAroundIdeographsInComplexText()
{
    return false;
}

bool FontCascade::canUseGlyphDisplayList(const RenderStyle&)
{
    return true;
}

ResolvedEmojiPolicy FontCascade::resolveEmojiPolicy(FontVariantEmoji fontVariantEmoji, char32_t character)
{
    switch (fontVariantEmoji) {
    case FontVariantEmoji::Normal:
    case FontVariantEmoji::Unicode:
        if (isEmojiWithPresentationByDefault(character)
            || isEmojiModifierBase(character)
            || isEmojiFitzpatrickModifier(character))
            return ResolvedEmojiPolicy::RequireEmoji;
        break;
    case FontVariantEmoji::Text:
        return ResolvedEmojiPolicy::RequireText;
    case FontVariantEmoji::Emoji:
        return ResolvedEmojiPolicy::RequireEmoji;
    }

    return ResolvedEmojiPolicy::NoPreference;
}

RefPtr<const Font> FontCascade::fontForCombiningCharacterSequence(StringView stringView) const
{
    ASSERT(!stringView.isEmpty());
    auto codePoints = stringView.codePoints();
    auto codePointsIterator = codePoints.begin();
    char32_t baseCharacter = *codePointsIterator;
    ++codePointsIterator;
    bool isOnlySingleCodePoint = codePointsIterator == codePoints.end();
    GlyphData baseCharacterGlyphData = glyphDataForCharacter(baseCharacter, false, NormalVariant);
    if (!baseCharacterGlyphData.glyph)
        return nullptr;

    if (isOnlySingleCodePoint)
        return baseCharacterGlyphData.font.get();

    bool triedBaseCharacterFont = false;
    for (unsigned i = 0; !fallbackRangesAt(i).isNull(); ++i) {
        auto& fontRanges = fallbackRangesAt(i);
        if (fontRanges.isGenericFontFamily() && isPrivateUseAreaCharacter(baseCharacter))
            continue;

        const Font* font = fontRanges.fontForCharacter(baseCharacter);
        if (!font)
            continue;

        if (font == baseCharacterGlyphData.font)
            triedBaseCharacterFont = true;

        if (font->canRenderCombiningCharacterSequence(stringView))
            return font;
    }

    if (!triedBaseCharacterFont && baseCharacterGlyphData.font && baseCharacterGlyphData.font->canRenderCombiningCharacterSequence(stringView))
        return baseCharacterGlyphData.font.get();
    return nullptr;
}

} // namespace WebCore

#endif // USE(SKIA)
