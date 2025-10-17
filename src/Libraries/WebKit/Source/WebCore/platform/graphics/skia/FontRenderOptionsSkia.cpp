/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#include "FontRenderOptions.h"

#if USE(SKIA)

namespace WebCore {

FontRenderOptions::FontRenderOptions() = default;

void FontRenderOptions::setHinting(std::optional<Hinting> hinting)
{
    switch (hinting.value_or(Hinting::Medium)) {
    case Hinting::None:
        m_hinting = SkFontHinting::kNone;
        break;
    case Hinting::Slight:
        m_hinting = SkFontHinting::kSlight;
        break;
    case Hinting::Medium:
        m_hinting = SkFontHinting::kNormal;
        break;
    case Hinting::Full:
        m_hinting = SkFontHinting::kFull;
        break;
    }
}

SkFontHinting FontRenderOptions::hinting() const
{
    if (m_isHintingDisabledForTesting)
        return SkFontHinting::kNone;

    if (m_followSystemSettings)
        return m_hinting;

    return SkFontHinting::kSlight;
}

void FontRenderOptions::setAntialias(std::optional<Antialias> antialias)
{
    switch (antialias.value_or(Antialias::Normal)) {
    case Antialias::None:
        m_antialias = SkFont::Edging::kAlias;
        break;
    case Antialias::Normal:
        m_antialias = SkFont::Edging::kAntiAlias;
        break;
    case Antialias::Subpixel:
        m_antialias = SkFont::Edging::kSubpixelAntiAlias;
        break;
    }
}

SkFont::Edging FontRenderOptions::antialias() const
{
    if (m_followSystemSettings)
        return m_antialias;

    return SkFont::Edging::kAntiAlias;
}

void FontRenderOptions::setSubpixelOrder(std::optional<SubpixelOrder> subpixelOrder)
{
    switch (subpixelOrder.value_or(SubpixelOrder::Unknown)) {
    case SubpixelOrder::Unknown:
        m_subpixelOrder = kUnknown_SkPixelGeometry;
        break;
    case SubpixelOrder::HorizontalRGB:
        m_subpixelOrder = kRGB_H_SkPixelGeometry;
        break;
    case SubpixelOrder::HorizontalBGR:
        m_subpixelOrder = kBGR_H_SkPixelGeometry;
        break;
    case SubpixelOrder::VerticalRGB:
        m_subpixelOrder = kRGB_V_SkPixelGeometry;
        break;
    case SubpixelOrder::VerticalBGR:
        m_subpixelOrder = kBGR_V_SkPixelGeometry;
        break;
    }
}

SkPixelGeometry FontRenderOptions::subpixelOrder() const
{
    if (m_followSystemSettings)
        return m_subpixelOrder;

    return kUnknown_SkPixelGeometry;
}

bool FontRenderOptions::useSubpixelPositioning() const
{
    // Subpixel positioning is not a system setting, it's set when device scale factor is >= 2.
    // So, we only apply it when not following system settings to force subpixel position with
    // linear metrics and no hinting.
    if (!m_followSystemSettings)
        return m_useSubpixelPositioning;

    return false;
}

void FontRenderOptions::disableHintingForTesting()
{
    m_isHintingDisabledForTesting = true;
}

} // namespace WebCore

#endif // USE(SKIA)
