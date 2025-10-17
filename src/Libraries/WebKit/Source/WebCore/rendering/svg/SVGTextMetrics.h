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
#pragma once

#include <wtf/text/WTFString.h>

namespace WebCore {

class RenderSVGInlineText;
class SVGTextLayoutAttributes;
class TextRun;

class SVGTextMetrics {
public:
    enum MetricsType { SkippedSpaceMetrics };

    SVGTextMetrics() = default;
    explicit SVGTextMetrics(MetricsType)
        : m_length(1)
    { }
    SVGTextMetrics(const RenderSVGInlineText&, unsigned length, float width);
    SVGTextMetrics(unsigned length, float scaledWidth, float scaledHeight)
        : m_width(scaledWidth)
        , m_height(scaledHeight)
        , m_length(length)
    { }

    static SVGTextMetrics measureCharacterRange(const RenderSVGInlineText&, unsigned position, unsigned length);
    static TextRun constructTextRun(const RenderSVGInlineText&, unsigned position = 0, unsigned length = std::numeric_limits<unsigned>::max());

    bool isEmpty() const { return !m_width && !m_height && !m_glyph.isValid && m_length == 1; }

    float width() const { return m_width; }
    void setWidth(float width) { m_width = width; }

    float height() const { return m_height; }
    unsigned length() const { return m_length; }

    struct Glyph {
        Glyph()
            : isValid(false)
        {
        }

        bool isValid;
        String name;
        String unicodeString;
    };

    // Only useful when measuring individual characters, to lookup ligatures.
    const Glyph& glyph() const { return m_glyph; }

private:
    SVGTextMetrics(const RenderSVGInlineText&, const TextRun&);

    float m_width { 0 };
    float m_height { 0 };
    unsigned m_length { 0 };
    Glyph m_glyph;
};

} // namespace WebCore
