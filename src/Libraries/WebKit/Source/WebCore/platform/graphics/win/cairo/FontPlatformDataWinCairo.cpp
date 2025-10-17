/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
#include "FontPlatformData.h"

#if PLATFORM(WIN) && USE(CAIRO)

#include "HWndDC.h"
#include "SharedBuffer.h"
#include <cairo-dwrite.h>
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

static IDWriteGdiInterop* getDWriteGdiInterop()
{
    static COMPtr<IDWriteGdiInterop> gdiInterop;
    if (gdiInterop)
        return gdiInterop.get();
    COMPtr<IDWriteFactory> factory;
    HRESULT hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&factory));
    RELEASE_ASSERT(SUCCEEDED(hr));
    hr = factory->GetGdiInterop(&gdiInterop);
    RELEASE_ASSERT(SUCCEEDED(hr));
    return gdiInterop.get();
}

cairo_font_face_t*
createCairoDWriteFontFace(HFONT font)
{
    IDWriteGdiInterop* gdiInterop = getDWriteGdiInterop();

    bool retry = false;
    GDIObject<HFONT> retryFont;
    COMPtr<IDWriteFontFace> dwFace;
    HWndDC hdc(nullptr);
    while (font) {
        HGDIOBJ oldFont = SelectObject(hdc, font);
        HRESULT hr = gdiInterop->CreateFontFaceFromHdc(hdc, &dwFace);
        SelectObject(hdc, oldFont);
        if (SUCCEEDED(hr))
            break;
        if (retry)
            break;
        // CreateFontFaceFromHdc may fail if the font size is too large. Retry it by creating a smaller font.
        LOGFONT logfont;
        GetObject(font, sizeof(logfont), &logfont);
        logfont.lfHeight = -32;
        retryFont = adoptGDIObject(CreateFontIndirect(&logfont));
        font = retryFont.get();
        retry = true;
    }
    RELEASE_ASSERT(dwFace);
    return cairo_dwrite_font_face_create_for_dwrite_fontface(dwFace.get());
}

void FontPlatformData::platformDataInit(HFONT font, float size)
{
    cairo_font_face_t* fontFace = createCairoDWriteFontFace(font);

    cairo_matrix_t sizeMatrix, ctm;
    cairo_matrix_init_identity(&ctm);
    cairo_matrix_init_scale(&sizeMatrix, size, size);

    static cairo_font_options_t* fontOptions = nullptr;
    if (!fontOptions) {
        fontOptions = cairo_font_options_create();
        cairo_font_options_set_antialias(fontOptions, CAIRO_ANTIALIAS_SUBPIXEL);
    }

    m_scaledFont = adoptRef(cairo_scaled_font_create(fontFace, &sizeMatrix, &ctm, fontOptions));
    cairo_font_face_destroy(fontFace);
}

FontPlatformData::FontPlatformData(GDIObject<HFONT> font, cairo_font_face_t* fontFace, float size, bool bold, bool oblique, const FontCustomPlatformData* customPlatformData)
    : FontPlatformData(size, bold, oblique, FontOrientation::Horizontal, FontWidthVariant::RegularWidth, TextRenderingMode::AutoTextRendering, customPlatformData)
{
    m_hfont = SharedGDIObject<HFONT>::create(WTFMove(font));

    cairo_matrix_t fontMatrix;
    cairo_matrix_init_scale(&fontMatrix, size, size);
    cairo_matrix_t ctm;
    cairo_matrix_init_identity(&ctm);
    cairo_font_options_t* options = cairo_font_options_create();

    // We force antialiasing and disable hinting to provide consistent
    // typographic qualities for custom fonts on all platforms.
    cairo_font_options_set_hint_style(options, CAIRO_HINT_STYLE_NONE);
    cairo_font_options_set_antialias(options, CAIRO_ANTIALIAS_BEST);

    if (syntheticOblique()) {
        static const float syntheticObliqueSkew = -tanf(14 * acosf(0) / 90);
        cairo_matrix_t skew = { 1, 0, syntheticObliqueSkew, 1, 0, 0 };
        cairo_matrix_multiply(&fontMatrix, &skew, &fontMatrix);
    }

    m_scaledFont = adoptRef(cairo_scaled_font_create(fontFace, &fontMatrix, &ctm, options));
    cairo_font_options_destroy(options);
}

unsigned FontPlatformData::hash() const
{
    return PtrHash<cairo_scaled_font_t*>::hash(m_scaledFont.get());
}

bool FontPlatformData::platformIsEqual(const FontPlatformData& other) const
{
    return m_hfont == other.m_hfont
        && m_scaledFont == other.m_scaledFont;
}

RefPtr<SharedBuffer> FontPlatformData::openTypeTable(uint32_t table) const
{
    return platformOpenTypeTable(table);
}

#if !LOG_DISABLED
String FontPlatformData::description() const
{
    return String();
}
#endif

String FontPlatformData::familyName() const
{
    HWndDC hdc(0);
    HGDIOBJ oldFont = SelectObject(hdc, m_hfont.get());
    wchar_t faceName[LF_FACESIZE];
    GetTextFace(hdc, LF_FACESIZE, faceName);
    SelectObject(hdc, oldFont);
    return faceName;
}

} // namespace WebCore

#endif // PLATFORM(WIN) && USE(CAIRO)
