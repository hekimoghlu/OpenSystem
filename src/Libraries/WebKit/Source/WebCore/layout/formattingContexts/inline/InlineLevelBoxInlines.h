/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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

#include "InlineLevelBox.h"
#include "RenderStyleInlines.h"

namespace WebCore {
namespace Layout {

inline InlineLevelBox::InlineLevelBox(const Box& layoutBox, const RenderStyle& style, InlineLayoutUnit logicalLeft, InlineLayoutSize logicalSize, Type type, OptionSet<PositionWithinLayoutBox> positionWithinLayoutBox)
    : m_layoutBox(layoutBox)
    , m_logicalRect({ }, logicalLeft, logicalSize.width(), logicalSize.height())
    , m_hasContent(layoutBox.isRubyBase() && layoutBox.associatedRubyAnnotationBox()) // Normally we set inline box's has-content state as we come across child content, but ruby annotations are not visible to inline layout.
    , m_isFirstWithinLayoutBox(positionWithinLayoutBox.contains(PositionWithinLayoutBox::First))
    , m_isLastWithinLayoutBox(positionWithinLayoutBox.contains(PositionWithinLayoutBox::Last))
    , m_type(type)
    , m_style({ style.fontCascade().metricsOfPrimaryFont(), style.lineHeight(), style.textBoxTrim(), style.textBoxEdge(), style.lineFitEdge(), style.lineBoxContain(), InlineLayoutUnit(style.fontCascade().fontDescription().computedSize()), { } })
{
    m_style.verticalAlignment.type = style.verticalAlign();
    if (m_style.verticalAlignment.type == VerticalAlign::Length)
        m_style.verticalAlignment.baselineOffset = floatValueForLength(style.verticalAlignLength(), preferredLineHeight());
}

inline InlineLevelBox InlineLevelBox::createAtomicInlineBox(const Box& layoutBox, const RenderStyle& style, InlineLayoutUnit logicalLeft, InlineLayoutUnit logicalWidth)
{
    return { layoutBox, style, logicalLeft, { logicalWidth, { } }, Type::AtomicInlineBox };
}

inline InlineLevelBox InlineLevelBox::createGenericInlineLevelBox(const Box& layoutBox, const RenderStyle& style, InlineLayoutUnit logicalLeft)
{
    return { layoutBox, style, logicalLeft, { }, Type::GenericInlineLevelBox };
}

inline InlineLevelBox InlineLevelBox::createInlineBox(const Box& layoutBox, const RenderStyle& style, InlineLayoutUnit logicalLeft, InlineLayoutUnit logicalWidth, LineSpanningInlineBox isLineSpanning)
{
    return { layoutBox, style, logicalLeft, { logicalWidth, { } }, isLineSpanning == LineSpanningInlineBox::Yes ? Type::LineSpanningInlineBox : Type::InlineBox, { } };
}

inline InlineLevelBox InlineLevelBox::createLineBreakBox(const Box& layoutBox, const RenderStyle& style, InlineLayoutUnit logicalLeft)
{
    return { layoutBox, style, logicalLeft, { }, Type::LineBreakBox };
}

inline InlineLevelBox InlineLevelBox::createRootInlineBox(const Box& layoutBox, const RenderStyle& style, InlineLayoutUnit logicalLeft, InlineLayoutUnit logicalWidth)
{
    return { layoutBox, style, logicalLeft, { logicalWidth, { } }, Type::RootInlineBox, { } };
}

inline bool InlineLevelBox::mayStretchLineBox() const
{
    if (isRootInlineBox())
        return m_style.lineBoxContain.containsAny({ LineBoxContain::Block, LineBoxContain::Inline }) || (hasContent() && m_style.lineBoxContain.containsAny({ LineBoxContain::InitialLetter, LineBoxContain::Font, LineBoxContain::Glyphs }));

    if (isAtomicInlineBox())
        return m_style.lineBoxContain.contains(LineBoxContain::Replaced);

    if (isInlineBox()) {
        // Either the inline box itself is included or its text content thorugh Glyph and Font.
        return m_style.lineBoxContain.containsAny({ LineBoxContain::Inline, LineBoxContain::InlineBox }) || (hasContent() && m_style.lineBoxContain.containsAny({ LineBoxContain::Font, LineBoxContain::Glyphs }));
    }

    if (isLineBreakBox())
        return m_style.lineBoxContain.containsAny({ LineBoxContain::Inline, LineBoxContain::InlineBox }) || (hasContent() && m_style.lineBoxContain.containsAny({ LineBoxContain::Font, LineBoxContain::Glyphs }));

    return true;
}

inline void InlineLevelBox::setTextEmphasis(std::pair<InlineLayoutUnit, InlineLayoutUnit> textEmphasis)
{
    if (!textEmphasis.first && !textEmphasis.second)
        return;
    if (textEmphasis.first) {
        m_textEmphasis = TextEmphasis { textEmphasis.first, 0.f };
        return;
    }
    m_textEmphasis = TextEmphasis { 0.f, textEmphasis.second };
}

}
}

