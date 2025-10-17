/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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

#include "AffineTransform.h"
#include "CairoOperations.h"
#include "CairoUtilities.h"
#include "Font.h"
#include "GlyphBuffer.h"
#include "Gradient.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "PathCairo.h"
#include "Pattern.h"
#include "RefPtrCairo.h"
#include "ShadowBlur.h"
#include <unicode/uchar.h>

namespace WebCore {

void FontCascade::drawGlyphs(GraphicsContext& context, const Font& font, std::span<const GlyphBufferGlyph> glyphs,
    std::span<const GlyphBufferAdvance> advances, const FloatPoint& point,
    FontSmoothingMode fontSmoothingMode)
{
    if (!font.platformData().size())
        return;

    auto xOffset = point.x();
    Vector<cairo_glyph_t> cairoGlyphs;
    cairoGlyphs.reserveInitialCapacity(glyphs.size());
    {
        auto yOffset = point.y();
        for (size_t i = 0; i < glyphs.size(); ++i) {
            bool append = true;
#if PLATFORM(WIN)
            // GlyphBuffer::makeGlyphInvisible expects 0xFFFF glyph is invisible. However, DirectWrite shows a blank square for it.
            append = glyphs[i] != 0xFFFF;
#endif
            if (append)
                cairoGlyphs.append({ glyphs[i], xOffset, yOffset });
            xOffset += advances[i].width();
            yOffset += advances[i].height();
        }
    }

    cairo_scaled_font_t* scaledFont = font.platformData().scaledFont();
    double syntheticBoldOffset = font.syntheticBoldOffset();

    if (!font.allowsAntialiasing())
        fontSmoothingMode = FontSmoothingMode::NoSmoothing;

    ASSERT(context.hasPlatformContext());
    auto& state = context.state();
    Cairo::drawGlyphs(*context.platformContext(), Cairo::FillSource(state), Cairo::StrokeSource(state),
        Cairo::ShadowState(state), point, scaledFont, syntheticBoldOffset, cairoGlyphs, xOffset,
        state.textDrawingMode(), state.strokeThickness(), state.dropShadow(), fontSmoothingMode);
}

Path Font::platformPathForGlyph(Glyph glyph) const
{
    RefPtr<cairo_t> cr = adoptRef(cairo_create(adoptRef(cairo_image_surface_create(CAIRO_FORMAT_A8, 1, 1)).get()));

    cairo_glyph_t cairoGlyph = { glyph, 0, 0 };
    cairo_set_scaled_font(cr.get(), platformData().scaledFont());
    cairo_glyph_path(cr.get(), &cairoGlyph, 1);

    float syntheticBoldOffset = this->syntheticBoldOffset();
    if (syntheticBoldOffset) {
        cairo_translate(cr.get(), syntheticBoldOffset, 0);
        cairo_glyph_path(cr.get(), &cairoGlyph, 1);
    }
    return { PathCairo::create(WTFMove(cr)) };
}

FloatRect Font::platformBoundsForGlyph(Glyph glyph) const
{
    if (!m_platformData.size())
        return FloatRect();

    cairo_glyph_t cglyph = { glyph, 0, 0 };
    cairo_text_extents_t extents;
    cairo_scaled_font_glyph_extents(m_platformData.scaledFont(), &cglyph, 1, &extents);

    if (cairo_scaled_font_status(m_platformData.scaledFont()) == CAIRO_STATUS_SUCCESS)
        return FloatRect(extents.x_bearing, extents.y_bearing, extents.width, extents.height);

    return FloatRect();
}

float Font::platformWidthForGlyph(Glyph glyph) const
{
    if (!m_platformData.size())
        return 0;

    if (cairo_scaled_font_status(m_platformData.scaledFont()) != CAIRO_STATUS_SUCCESS)
        return m_spaceWidth;

    cairo_glyph_t cairoGlyph = { glyph, 0, 0 };
    cairo_text_extents_t extents;
    cairo_scaled_font_glyph_extents(m_platformData.scaledFont(), &cairoGlyph, 1, &extents);
    float width = platformData().orientation() == FontOrientation::Horizontal ? extents.x_advance : -extents.y_advance;
    return width ? width : m_spaceWidth;
}

ResolvedEmojiPolicy FontCascade::resolveEmojiPolicy(FontVariantEmoji fontVariantEmoji, char32_t)
{
    // FIXME: https://bugs.webkit.org/show_bug.cgi?id=259205 We can't return RequireText or RequireEmoji
    // unless we have a way of knowing whether a font/glyph is color or not.
    switch (fontVariantEmoji) {
    case FontVariantEmoji::Normal:
    case FontVariantEmoji::Unicode:
        return ResolvedEmojiPolicy::NoPreference;
    case FontVariantEmoji::Text:
        return ResolvedEmojiPolicy::RequireText;
    case FontVariantEmoji::Emoji:
        return ResolvedEmojiPolicy::RequireEmoji;
    }
    return ResolvedEmojiPolicy::NoPreference;
}

} // namespace WebCore

#endif // USE(CAIRO)
