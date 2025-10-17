/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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

#if USE(CAIRO)

#include "FontCache.h"
#include "SurrogatePairAwareTextIterator.h"
#include <wtf/text/CharacterProperties.h>

namespace WebCore {

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

static bool characterSequenceIsEmoji(SurrogatePairAwareTextIterator& iterator, char32_t firstCharacter, unsigned firstClusterLength)
{
    char32_t character = firstCharacter;
    unsigned clusterLength = firstClusterLength;
    if (!iterator.consume(character, clusterLength))
        return false;

    if (isEmojiKeycapBase(character)) {
        iterator.advance(clusterLength);
        char32_t nextCharacter;
        if (!iterator.consume(nextCharacter, clusterLength))
            return false;

        if (nextCharacter == combiningEnclosingKeycap)
            return true;

        // Variation selector 16.
        if (nextCharacter == 0xFE0F) {
            iterator.advance(clusterLength);
            if (!iterator.consume(nextCharacter, clusterLength))
                return false;

            if (nextCharacter == combiningEnclosingKeycap)
                return true;
        }

        return false;
    }

    // Regional indicator.
    if (isEmojiRegionalIndicator(character)) {
        iterator.advance(clusterLength);
        char32_t nextCharacter;
        if (!iterator.consume(nextCharacter, clusterLength))
            return false;

        if (isEmojiRegionalIndicator(nextCharacter))
            return true;

        return false;
    }

    if (character == combiningEnclosingKeycap)
        return true;

    if (isEmojiWithPresentationByDefault(character)
        || isEmojiModifierBase(character)
        || isEmojiFitzpatrickModifier(character))
        return true;

    return false;
}

RefPtr<const Font> FontCascade::fontForCombiningCharacterSequence(StringView stringView) const
{
    auto normalizedString = normalizedNFC(stringView);

    // Code below relies on normalizedNFC never narrowing a 16-bit input string into an 8-bit output string.
    // At the time of this writing, the function never does this, but in theory a future version could, and
    // we would then need to add code paths here for the simpler 8-bit case.
    auto characters = normalizedString.view.span16();
    auto length = normalizedString.view.length();

    char32_t character;
    unsigned clusterLength = 0;
    SurrogatePairAwareTextIterator iterator(characters, 0, length);
    if (!iterator.consume(character, clusterLength))
        return nullptr;

    bool isEmoji = characterSequenceIsEmoji(iterator, character, clusterLength);
    bool preferColoredFont = isEmoji;
    // U+FE0E forces text style.
    // U+FE0F forces emoji style.
    if (characters[length - 1] == 0xFE0E)
        preferColoredFont = false;
    else if (characters[length - 1] == 0xFE0F)
        preferColoredFont = true;

    RefPtr baseFont = glyphDataForCharacter(character, false, NormalVariant).font.get();
    if (baseFont
        && (clusterLength == length || baseFont->canRenderCombiningCharacterSequence(normalizedString.view))
        && (!preferColoredFont || baseFont->platformData().isColorBitmapFont()))
        return baseFont.get();

    for (unsigned i = 0; !fallbackRangesAt(i).isNull(); ++i) {
        RefPtr fallbackFont = fallbackRangesAt(i).fontForCharacter(character);
        if (!fallbackFont || fallbackFont == baseFont)
            continue;

        if (fallbackFont->canRenderCombiningCharacterSequence(normalizedString.view) && (!preferColoredFont || fallbackFont->platformData().isColorBitmapFont()))
            return fallbackFont.get();
    }

    const auto& originalFont = fallbackRangesAt(0).fontForFirstRange();
    if (auto systemFallback = FontCache::forCurrentThread().systemFallbackForCharacterCluster(m_fontDescription, originalFont, IsForPlatformFont::No, preferColoredFont ? FontCache::PreferColoredFont::Yes : FontCache::PreferColoredFont::No, normalizedString.view)) {
        if (systemFallback->canRenderCombiningCharacterSequence(normalizedString.view) && (!preferColoredFont || systemFallback->platformData().isColorBitmapFont()))
            return systemFallback.get();

        // In case of emoji, if fallback font is colored try again without the variation selector character.
        if (isEmoji && characters[length - 1] == 0xFE0F && systemFallback->platformData().isColorBitmapFont() && systemFallback->canRenderCombiningCharacterSequence(characters.first(length - 1)))
            return systemFallback.get();
    }

    return baseFont.get();
}

} // namespace WebCore

#endif // USE(CAIRO)
