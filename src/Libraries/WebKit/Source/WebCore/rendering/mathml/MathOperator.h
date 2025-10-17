/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#if ENABLE(MATHML)

#include "GlyphPage.h"
#include "LayoutUnit.h"
#include "OpenTypeMathData.h"
#include "PaintInfo.h"
#include <unicode/utypes.h>

namespace WebCore {

class RenderStyle;

class MathOperator {
public:
    MathOperator();
    enum class Type { NormalOperator, DisplayOperator, VerticalOperator, HorizontalOperator };
    void setOperator(const RenderStyle&, char32_t baseCharacter, Type);
    void reset(const RenderStyle&);

    LayoutUnit width() const { return m_width; }
    LayoutUnit maxPreferredWidth() const { return m_maxPreferredWidth; }
    LayoutUnit ascent() const { return m_ascent; }
    LayoutUnit descent() const { return m_descent; }
    LayoutUnit italicCorrection() const { return m_italicCorrection; }

    void stretchTo(const RenderStyle&, LayoutUnit width);

    void paint(const RenderStyle&, PaintInfo&, const LayoutPoint&);

private:
    struct GlyphAssemblyData {
        char32_t topOrRightCodePoint { 0 };
        Glyph topOrRightFallbackGlyph { 0 };
        char32_t extensionCodePoint { 0 };
        Glyph extensionFallbackGlyph { 0 };
        char32_t bottomOrLeftCodePoint { 0 };
        Glyph bottomOrLeftFallbackGlyph { 0 };
        char32_t middleCodePoint { 0 };
        Glyph middleFallbackGlyph { 0 };

        bool hasExtension() const { return extensionCodePoint || extensionFallbackGlyph; }
        bool hasMiddle() const { return middleCodePoint || middleFallbackGlyph; }
        void initialize();
    };
    enum class StretchType { Unstretched, SizeVariant, GlyphAssembly };
    enum GlyphPaintTrimming {
        TrimTop,
        TrimBottom,
        TrimTopAndBottom,
        TrimLeft,
        TrimRight,
        TrimLeftAndRight
    };

    LayoutUnit stretchSize() const;
    bool getGlyph(const RenderStyle&, char32_t character, GlyphData&) const;
    bool getBaseGlyph(const RenderStyle& style, GlyphData& baseGlyph) const { return getGlyph(style, m_baseCharacter, baseGlyph); }
    void setSizeVariant(const GlyphData&);
    void setGlyphAssembly(const RenderStyle&, const GlyphAssemblyData&);
    void getMathVariantsWithFallback(const RenderStyle&, bool isVertical, Vector<Glyph>&, Vector<OpenTypeMathData::AssemblyPart>&);
    void calculateDisplayStyleLargeOperator(const RenderStyle&);
    void calculateStretchyData(const RenderStyle&, bool calculateMaxPreferredWidth, LayoutUnit targetSize = 0_lu);
    bool calculateGlyphAssemblyFallback(const Vector<OpenTypeMathData::AssemblyPart>&, GlyphAssemblyData&) const;

    LayoutRect paintGlyph(const RenderStyle&, PaintInfo&, const GlyphData&, const LayoutPoint& origin, GlyphPaintTrimming);
    void fillWithVerticalExtensionGlyph(const RenderStyle&, PaintInfo&, const LayoutPoint& from, const LayoutPoint& to);
    void fillWithHorizontalExtensionGlyph(const RenderStyle&, PaintInfo&, const LayoutPoint& from, const LayoutPoint& to);
    void paintVerticalGlyphAssembly(const RenderStyle&, PaintInfo&, const LayoutPoint&);
    void paintHorizontalGlyphAssembly(const RenderStyle&, PaintInfo&, const LayoutPoint&);

    char32_t m_baseCharacter { 0 };
    Type m_operatorType { Type::NormalOperator };
    StretchType m_stretchType { StretchType::Unstretched };
    union {
        Glyph m_variantGlyph;
        GlyphAssemblyData m_assembly;
    };
    LayoutUnit m_maxPreferredWidth { 0 };
    LayoutUnit m_width { 0 };
    LayoutUnit m_ascent { 0 };
    LayoutUnit m_descent { 0 };
    LayoutUnit m_italicCorrection { 0 };
    float m_radicalVerticalScale { 1 };
};

}

#endif // ENABLE(MATHML)
