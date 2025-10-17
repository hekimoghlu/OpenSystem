/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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

#if USE(CAIRO)

namespace WebCore {

FontRenderOptions::FontRenderOptions()
    : m_fontOptions(cairo_font_options_create())
{
}

void FontRenderOptions::setHinting(std::optional<Hinting> hinting)
{
    if (m_isHintingDisabledForTesting)
        return;

    cairo_font_options_set_hint_metrics(m_fontOptions.get(), CAIRO_HINT_METRICS_ON);
    if (!hinting.has_value()) {
        cairo_font_options_set_hint_style(m_fontOptions.get(), CAIRO_HINT_STYLE_DEFAULT);
        return;
    }

    switch (hinting.value()) {
    case Hinting::None:
        cairo_font_options_set_hint_style(m_fontOptions.get(), CAIRO_HINT_STYLE_NONE);
        break;
    case Hinting::Slight:
        cairo_font_options_set_hint_style(m_fontOptions.get(), CAIRO_HINT_STYLE_SLIGHT);
        break;
    case Hinting::Medium:
        cairo_font_options_set_hint_style(m_fontOptions.get(), CAIRO_HINT_STYLE_MEDIUM);
        break;
    case Hinting::Full:
        cairo_font_options_set_hint_style(m_fontOptions.get(), CAIRO_HINT_STYLE_FULL);
        break;
    }
}

void FontRenderOptions::setAntialias(std::optional<Antialias> antialias)
{
    if (!antialias.has_value()) {
        cairo_font_options_set_antialias(m_fontOptions.get(), CAIRO_ANTIALIAS_DEFAULT);
        return;
    }

    switch (antialias.value()) {
    case Antialias::None:
        cairo_font_options_set_antialias(m_fontOptions.get(), CAIRO_ANTIALIAS_NONE);
        break;
    case Antialias::Normal:
        cairo_font_options_set_antialias(m_fontOptions.get(), CAIRO_ANTIALIAS_GRAY);
        break;
    case Antialias::Subpixel:
        cairo_font_options_set_antialias(m_fontOptions.get(), CAIRO_ANTIALIAS_SUBPIXEL);
        break;
    }
}

void FontRenderOptions::setSubpixelOrder(std::optional<SubpixelOrder> subpixelOrder)
{
    if (!subpixelOrder.has_value()) {
        cairo_font_options_set_subpixel_order(m_fontOptions.get(), CAIRO_SUBPIXEL_ORDER_DEFAULT);
        return;
    }

    switch (subpixelOrder.value()) {
    case SubpixelOrder::Unknown:
        cairo_font_options_set_subpixel_order(m_fontOptions.get(), CAIRO_SUBPIXEL_ORDER_DEFAULT);
        break;
    case SubpixelOrder::HorizontalRGB:
        cairo_font_options_set_subpixel_order(m_fontOptions.get(), CAIRO_SUBPIXEL_ORDER_RGB);
        break;
    case SubpixelOrder::HorizontalBGR:
        cairo_font_options_set_subpixel_order(m_fontOptions.get(), CAIRO_SUBPIXEL_ORDER_BGR);
        break;
    case SubpixelOrder::VerticalRGB:
        cairo_font_options_set_subpixel_order(m_fontOptions.get(), CAIRO_SUBPIXEL_ORDER_VRGB);
        break;
    case SubpixelOrder::VerticalBGR:
        cairo_font_options_set_subpixel_order(m_fontOptions.get(), CAIRO_SUBPIXEL_ORDER_VBGR);
        break;
    }
}

void FontRenderOptions::disableHintingForTesting()
{
    cairo_font_options_set_hint_metrics(m_fontOptions.get(), CAIRO_HINT_METRICS_ON);
    cairo_font_options_set_hint_style(m_fontOptions.get(), CAIRO_HINT_STYLE_NONE);
    m_isHintingDisabledForTesting = true;
}

} // namespace WebCore

#endif // USE(CAIRO)
