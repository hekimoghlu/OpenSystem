/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#include "HWndDC.h"
#include <wtf/text/win/WCharStringExtras.h>

namespace WebCore {

bool GlyphPage::fill(std::span<const UChar> buffer)
{
    ASSERT(buffer.size() == GlyphPage::size || buffer.size() == 2 * GlyphPage::size);

    const Font& font = this->font();
    bool haveGlyphs = false;

    HWndDC dc(0);
    SaveDC(dc);
    SelectObject(dc, font.platformData().hfont());

    // FIXME: https://bugs.webkit.org/show_bug.cgi?id=259205 Determine if the glyph is a color glyph or not.
    if (buffer.size() == GlyphPage::size) {
        WORD localGlyphBuffer[GlyphPage::size * 2];
        DWORD result = GetGlyphIndices(dc, wcharFrom(buffer.data()), buffer.size(), localGlyphBuffer, GGI_MARK_NONEXISTING_GLYPHS);
        bool success = result != GDI_ERROR && static_cast<unsigned>(result) == buffer.size();

        if (success) {
            for (unsigned i = 0; i < GlyphPage::size; i++) {
                Glyph glyph = localGlyphBuffer[i];
                if (glyph == 0xffff)
                    setGlyphForIndex(i, 0, ColorGlyphType::Outline);
                else {
                    setGlyphForIndex(i, glyph, font.colorGlyphType(glyph));
                    haveGlyphs = true;
                }
            }
        }
    } else {
        SCRIPT_CACHE sc = { };
        SCRIPT_FONTPROPERTIES fp = { };
        fp.cBytes = sizeof fp;
        ScriptGetFontProperties(dc, &sc, &fp);
        ScriptFreeCache(&sc);

        for (unsigned i = 0; i < GlyphPage::size; i++) {
            wchar_t glyphs[2] = { };
            GCP_RESULTS gcpResults = { };
            gcpResults.lStructSize = sizeof gcpResults;
            gcpResults.nGlyphs = 2;
            gcpResults.lpGlyphs = glyphs;
            GetCharacterPlacement(dc, wcharFrom(buffer.data()) + i * 2, 2, 0, &gcpResults, GCP_GLYPHSHAPE);
            bool success = 1 == gcpResults.nGlyphs;
            if (success) {
                auto glyph = glyphs[0];
                if (glyph == fp.wgBlank || glyph == fp.wgInvalid || glyph == fp.wgDefault)
                    setGlyphForIndex(i, 0, ColorGlyphType::Outline);
                else {
                    setGlyphForIndex(i, glyph, font.colorGlyphType(glyph));
                    haveGlyphs = true;
                }
            }
        }
    }
    RestoreDC(dc, -1);

    return haveGlyphs;
}

}
