/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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

#include "CairoUtilities.h"
#include "Font.h"
#include "FontCascade.h"
#include <cairo-ft.h>
#include <cairo.h>
#include <fontconfig/fcfreetype.h>
#include <wtf/text/CharacterProperties.h>

namespace WebCore {

bool GlyphPage::fill(std::span<const UChar> buffer)
{
    const Font& font = this->font();
    cairo_scaled_font_t* scaledFont = font.platformData().scaledFont();
    ASSERT(scaledFont);

    CairoFtFaceLocker cairoFtFaceLocker(scaledFont);
    FT_Face face = cairoFtFaceLocker.ftFace();
    if (!face)
        return false;

    std::optional<Glyph> zeroWidthSpaceGlyphValue;
    auto zeroWidthSpaceGlyph =
        [&] {
            if (!zeroWidthSpaceGlyphValue)
                zeroWidthSpaceGlyphValue = FcFreeTypeCharIndex(face, zeroWidthSpace);
            return *zeroWidthSpaceGlyphValue;
        };

    bool haveGlyphs = false;
    unsigned bufferOffset = 0;
    for (unsigned i = 0; i < GlyphPage::size; i++) {
        if (bufferOffset == buffer.size())
            break;
        char32_t character;
        U16_NEXT(buffer, bufferOffset, buffer.size(), character);

        Glyph glyph = FcFreeTypeCharIndex(face, FontCascade::treatAsSpace(character) ? space : character);
        // If the font doesn't support a Default_Ignorable character, replace it with zero with space.
        if (!glyph && (isDefaultIgnorableCodePoint(character) || isControlCharacter(character)))
            glyph = zeroWidthSpaceGlyph();

        // FIXME: https://bugs.webkit.org/show_bug.cgi?id=259205 Determine if the glyph is a color glyph or not.
        if (!glyph)
            setGlyphForIndex(i, 0, ColorGlyphType::Outline);
        else {
            setGlyphForIndex(i, glyph, font.colorGlyphType(glyph));
            haveGlyphs = true;
        }
    }

    return haveGlyphs;
}

}
