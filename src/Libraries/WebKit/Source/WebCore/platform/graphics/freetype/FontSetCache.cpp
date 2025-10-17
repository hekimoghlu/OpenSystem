/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#include "FontSetCache.h"

#include "CairoUtilities.h"
#include "FontCache.h"
#include <wtf/text/CharacterProperties.h>

namespace WebCore {

FontSetCache::FontSet::FontSet(RefPtr<FcPattern>&& fontPattern)
    : pattern(WTFMove(fontPattern))
{
    FcResult result;
    fontSet.reset(FcFontSort(nullptr, pattern.get(), FcTrue, nullptr, &result));
    for (int i = 0; i < fontSet->nfont; ++i) {
        FcPattern* fontSetPattern = fontSet->fonts[i];
        FcCharSet* charSet;

        if (FcPatternGetCharSet(fontSetPattern, FC_CHARSET, 0, &charSet) == FcResultMatch)
            patterns.append({ fontSetPattern, charSet });
    }
}

RefPtr<FcPattern> FontSetCache::bestForCharacters(const FontDescription& fontDescription, bool preferColoredFont, StringView stringView)
{
    auto addResult = m_cache.ensure(FontSetCacheKey(fontDescription, preferColoredFont), [&fontDescription, preferColoredFont]() -> std::unique_ptr<FontSetCache::FontSet> {
        RefPtr<FcPattern> pattern = adoptRef(FcPatternCreate());
        FcPatternAddBool(pattern.get(), FC_SCALABLE, FcTrue);
#ifdef FC_COLOR
        if (preferColoredFont)
            FcPatternAddBool(pattern.get(), FC_COLOR, FcTrue);
#else
        UNUSED_VARIABLE(preferColoredFont);
#endif
        if (!FontCache::configurePatternForFontDescription(pattern.get(), fontDescription))
            return nullptr;

        FcConfigSubstitute(nullptr, pattern.get(), FcMatchPattern);
        cairo_ft_font_options_substitute(getDefaultCairoFontOptions(), pattern.get());
        FcDefaultSubstitute(pattern.get());
        return makeUnique<FontSetCache::FontSet>(WTFMove(pattern));
    });

    if (!addResult.iterator->value)
        return nullptr;

    auto& cachedFontSet = *addResult.iterator->value;
    if (cachedFontSet.patterns.isEmpty()) {
        FcResult result;
        return adoptRef(FcFontMatch(nullptr, cachedFontSet.pattern.get(), &result));
    }

    FcUniquePtr<FcCharSet> fontConfigCharSet(FcCharSetCreate());
    bool hasNonIgnorableCharacters = false;
    for (char32_t character : stringView.codePoints()) {
        if (!isDefaultIgnorableCodePoint(character)) {
            FcCharSetAddChar(fontConfigCharSet.get(), character);
            hasNonIgnorableCharacters = true;
        }
    }

    FcPattern* bestPattern = nullptr;
    if (hasNonIgnorableCharacters) {
        int minScore = std::numeric_limits<int>::max();
        for (const auto& [pattern, charSet] : cachedFontSet.patterns) {
            if (!charSet)
                continue;

            int score = FcCharSetSubtractCount(fontConfigCharSet.get(), charSet);
            if (!score) {
                bestPattern = pattern;
                break;
            }

            if (score < minScore) {
                bestPattern = pattern;
                minScore = score;
            }
        }
    }

    // If there aren't fonts with the given characters or all characters are ignorable, the first one is the best match.
    if (!bestPattern)
        bestPattern = cachedFontSet.patterns[0].first;

    return adoptRef(FcFontRenderPrepare(nullptr, cachedFontSet.pattern.get(), bestPattern));
}

void FontSetCache::clear()
{
    m_cache.clear();
}

} // namespace WebCore
