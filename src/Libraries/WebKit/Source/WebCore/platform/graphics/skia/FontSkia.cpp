/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#include "Font.h"

#include "GlyphBuffer.h"
#include "NotImplemented.h"
#include "PathSkia.h"
#include <skia/core/SkFont.h>
#include <skia/core/SkFontMetrics.h>

namespace WebCore {

Path Font::platformPathForGlyph(Glyph glyph) const
{
    auto path = PathSkia::create();
    const auto& font = m_platformData.skFont();
    font.getPath(glyph, path->platformPath());
    return { path };
}

FloatRect Font::platformBoundsForGlyph(Glyph glyph) const
{
    if (!m_platformData.size())
        return { };

    const auto& font = m_platformData.skFont();
    SkRect bounds;
    font.getBounds(&glyph, 1, &bounds, nullptr);
    if (!font.isSubpixel()) {
        SkIRect rect;
        bounds.roundOut(&rect);
        bounds.set(rect);
    }
    return bounds;
}

float Font::platformWidthForGlyph(Glyph glyph) const
{
    if (!m_platformData.size())
        return 0;

    const auto& font = m_platformData.skFont();
    SkScalar width;
    font.getWidths(&glyph, 1, &width);

    if (!font.isSubpixel())
        width = SkScalarRoundToInt(width);

    return SkScalarToFloat(width);
}

void Font::platformInit()
{
    if (!m_platformData.size())
        return;

    const auto& font = m_platformData.skFont();
    SkFontMetrics metrics;
    font.getMetrics(&metrics);

    auto ascent = SkScalarRoundToScalar(-metrics.fAscent);
    auto descent = SkScalarRoundToScalar(metrics.fDescent);
    m_fontMetrics.setAscent(ascent);
    m_fontMetrics.setDescent(descent);

    auto lineGap = SkScalarToFloat(metrics.fLeading);
    m_fontMetrics.setLineGap(lineGap);
    m_fontMetrics.setLineSpacing(lroundf(ascent) + lroundf(descent) + lroundf(lineGap));

    m_fontMetrics.setCapHeight(metrics.fCapHeight);

    float underlinePosition;
    if (metrics.hasUnderlinePosition(&underlinePosition))
        m_fontMetrics.setUnderlinePosition(underlinePosition);
    float underlineThickness;
    if (metrics.hasUnderlineThickness(&underlineThickness))
        m_fontMetrics.setUnderlineThickness(underlineThickness);

    if (metrics.fXHeight)
        m_fontMetrics.setXHeight(metrics.fXHeight);

    m_maxCharWidth = SkScalarRoundToInt(metrics.fXMax - metrics.fXMin);
    if (metrics.fAvgCharWidth)
        m_avgCharWidth = SkScalarToFloat(metrics.fAvgCharWidth);

    m_fontMetrics.setUnitsPerEm(font.getTypeface()->getUnitsPerEm());

    SkString familyName;
    font.getTypeface()->getFamilyName(&familyName);
    if (equalIgnoringASCIICase(familyName.c_str(), "Ahem"_s))
        m_allowsAntialiasing = false;
}

void Font::platformCharWidthInit()
{
    m_avgCharWidth = 0.f;
    m_maxCharWidth = 0.f;
    initCharWidths();
}

RefPtr<Font> Font::platformCreateScaledFont(const FontDescription&, float scaleFactor) const
{
    return Font::create(FontPlatformData(m_platformData.skFont().refTypeface(), scaleFactor * m_platformData.size(),
        m_platformData.syntheticBold(),
        m_platformData.syntheticOblique(),
        m_platformData.orientation(),
        m_platformData.widthVariant(),
        m_platformData.textRenderingMode(),
        Vector<hb_feature_t> { m_platformData.features() },
        m_platformData.customPlatformData()),
        origin(), IsInterstitial::No);
}

RefPtr<Font> Font::platformCreateHalfWidthFont() const
{
    // FIXME: https://bugs.webkit.org/show_bug.cgi?id=281333 : implement half width font for this platform.
    notImplemented();
    return nullptr;
}

void Font::determinePitch()
{
    m_treatAsFixedPitch = m_platformData.isFixedPitch();
}

bool Font::variantCapsSupportedForSynthesis(FontVariantCaps fontVariantCaps) const
{
    switch (fontVariantCaps) {
    case FontVariantCaps::Small:
    case FontVariantCaps::Petite:
    case FontVariantCaps::AllSmall:
    case FontVariantCaps::AllPetite:
        return false;
    default:
        // Synthesis only supports the variant-caps values listed above.
        return true;
    }
}

bool Font::platformSupportsCodePoint(char32_t character, std::optional<char32_t>) const
{
    return m_platformData.skFont().getTypeface()->unicharToGlyph(character);
}

} // namespace WebCore
