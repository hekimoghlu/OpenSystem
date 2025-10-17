/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#include "FontCache.h"

#if PLATFORM(WIN) && USE(CAIRO)

namespace WebCore {

LONG adjustedGDIFontWeight(LONG, const String&);
GDIObject<HFONT> createGDIFont(const AtomString&, LONG, bool, int);
LONG toGDIFontWeight(FontSelectionValue);
bool isGDIFontWeightBold(LONG);

std::unique_ptr<FontPlatformData> FontCache::createFontPlatformData(const FontDescription& fontDescription, const AtomString& family, const FontCreationContext&, OptionSet<FontLookupOptions> options)
{
    LONG weight = adjustedGDIFontWeight(toGDIFontWeight(fontDescription.weight()), family);
    auto hfont = createGDIFont(family, weight, isItalic(fontDescription.italic()),
        fontDescription.computedSize() * cWindowsFontScaleFactor);

    if (!hfont)
        return nullptr;

    LOGFONT logFont;
    GetObject(hfont.get(), sizeof(LOGFONT), &logFont);

    bool synthesizeBold = !options.contains(FontLookupOptions::DisallowBoldSynthesis)
        && isGDIFontWeightBold(weight) && !isGDIFontWeightBold(logFont.lfWeight);
    bool synthesizeItalic = !options.contains(FontLookupOptions::DisallowObliqueSynthesis)
        && isItalic(fontDescription.italic()) && !logFont.lfItalic;

    auto result = makeUnique<FontPlatformData>(WTFMove(hfont), fontDescription.computedSize(), synthesizeBold, synthesizeItalic);

    bool fontCreationFailed = !result->scaledFont();

    if (fontCreationFailed) {
        // The creation of the cairo scaled font failed for some reason. We already asserted in debug builds, but to make
        // absolutely sure that we don't use this font, go ahead and return 0 so that we can fall back to the next
        // font.
        return nullptr;
    }

    return result;
}

} // namespace WebCore

#endif // PLATFORM(WIN) && USE(CAIRO)
