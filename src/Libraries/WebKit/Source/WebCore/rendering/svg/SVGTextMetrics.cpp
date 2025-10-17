/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#include "SVGTextMetrics.h"

#include "RenderSVGInlineText.h"
#include "UnicodeBidi.h"

namespace WebCore {

SVGTextMetrics::SVGTextMetrics(const RenderSVGInlineText& textRenderer, const TextRun& run)
{
    float scalingFactor = textRenderer.scalingFactor();
    ASSERT(scalingFactor);

    const FontCascade& scaledFont = textRenderer.scaledFont();

    // Calculate width/height using the scaled font, divide this result by the scalingFactor afterwards.
    m_width = scaledFont.width(run) / scalingFactor;
    m_glyph.name = emptyString();
    m_height = scaledFont.metricsOfPrimaryFont().height() / scalingFactor;

    m_glyph.unicodeString = run.text().toString();
    m_glyph.isValid = true;

    m_length = static_cast<unsigned>(run.length());
}

TextRun SVGTextMetrics::constructTextRun(const RenderSVGInlineText& text, unsigned position, unsigned length)
{
    const RenderStyle& style = text.style();

    TextRun run(StringView(text.text()).substring(position, length),
        0, /* xPos, only relevant with allowTabs=true */
        0, /* padding, only relevant for justified text, not relevant for SVG */
        ExpansionBehavior::allowRightOnly(),
        style.writingMode().bidiDirection(),
        isOverride(style.unicodeBidi()) /* directionalOverride */);

    // We handle letter & word spacing ourselves.
    run.disableSpacing();
    return run;
}

SVGTextMetrics SVGTextMetrics::measureCharacterRange(const RenderSVGInlineText& text, unsigned position, unsigned length)
{
    return SVGTextMetrics(text, constructTextRun(text, position, length));
}

SVGTextMetrics::SVGTextMetrics(const RenderSVGInlineText& text, unsigned length, float width)
{
    float scalingFactor = text.scalingFactor();
    ASSERT(scalingFactor);

    m_width = width / scalingFactor;
    m_height = text.scaledFont().metricsOfPrimaryFont().height() / scalingFactor;

    m_length = length;
}

}
