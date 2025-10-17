/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
#include "FontCustomPlatformData.h"

#if PLATFORM(WIN) && USE(CAIRO)

#include "FontDescription.h"
#include <cairo-win32.h>

namespace WebCore {

cairo_font_face_t* createCairoDWriteFontFace(HFONT);

FontPlatformData FontCustomPlatformData::fontPlatformData(const FontDescription& fontDescription, bool bold, bool italic, const FontCreationContext&)
{
    auto size = fontDescription.computedSize();

    LOGFONT logFont;
    memset(&logFont, 0, sizeof(LOGFONT));
    wcsncpy(logFont.lfFaceName, name.wideCharacters().data(), LF_FACESIZE - 1);

    logFont.lfHeight = -size * cWindowsFontScaleFactor;
    logFont.lfWidth = 0;
    logFont.lfEscapement = 0;
    logFont.lfOrientation = 0;
    logFont.lfUnderline = false;
    logFont.lfStrikeOut = false;
    logFont.lfCharSet = DEFAULT_CHARSET;
    logFont.lfOutPrecision = OUT_TT_ONLY_PRECIS;
    logFont.lfQuality = CLEARTYPE_QUALITY;
    logFont.lfPitchAndFamily = DEFAULT_PITCH | FF_DONTCARE;
    logFont.lfItalic = italic;
    logFont.lfWeight = bold ? 700 : 400;

    auto hfont = adoptGDIObject(::CreateFontIndirect(&logFont));

    cairo_font_face_t* fontFace = createCairoDWriteFontFace(hfont.get());

    FontPlatformData fontPlatformData(WTFMove(hfont), fontFace, size, bold, italic, this);

    cairo_font_face_destroy(fontFace);

    return fontPlatformData;
}

} // namespace WebCore

#endif // PLATFORM(WIN) && USE(CAIRO)
