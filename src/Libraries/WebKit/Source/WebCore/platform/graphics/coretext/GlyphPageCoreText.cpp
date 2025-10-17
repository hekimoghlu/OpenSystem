/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
#include "GlyphPage.h"

#include "Font.h"
#include "FontCascade.h"
#include <pal/spi/cf/CoreTextSPI.h>
#include <pal/spi/cg/CoreGraphicsSPI.h>

namespace WebCore {

static bool shouldFillWithVerticalGlyphs(std::span<const UChar> buffer, const Font& font)
{
    if (!font.hasVerticalGlyphs())
        return false;
    for (auto character : buffer) {
        if (!FontCascade::isCJKIdeograph(character))
            return true;
    }
    return false;
}


bool GlyphPage::fill(std::span<const UChar> buffer)
{
    ASSERT(buffer.size() == GlyphPage::size || buffer.size() == 2 * GlyphPage::size);

    const Font& font = this->font();
    Vector<CGGlyph, 512> glyphs(buffer.size());
    unsigned glyphStep = buffer.size() / GlyphPage::size;

    if (shouldFillWithVerticalGlyphs(buffer, font))
        CTFontGetVerticalGlyphsForCharacters(font.platformData().ctFont(), reinterpret_cast<const UniChar*>(buffer.data()), glyphs.data(), buffer.size());
    else
        CTFontGetGlyphsForCharacters(font.platformData().ctFont(), reinterpret_cast<const UniChar*>(buffer.data()), glyphs.data(), buffer.size());

    bool haveGlyphs = false;
    for (unsigned i = 0; i < GlyphPage::size; ++i) {
        auto theGlyph = glyphs[i * glyphStep];
        if (theGlyph && theGlyph != deletedGlyph) {
            setGlyphForIndex(i, theGlyph, font.colorGlyphType(theGlyph));
            haveGlyphs = true;
        }
    }
    return haveGlyphs;
}

} // namespace WebCore
