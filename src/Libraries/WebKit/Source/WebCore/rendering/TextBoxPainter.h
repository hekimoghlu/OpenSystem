/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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

#include "FloatRect.h"
#include "FloatRoundedRect.h"
#include "InlineIteratorInlineBox.h"
#include "InlineIteratorTextBox.h"
#include "RenderObject.h"
#include "TextBoxSelectableRange.h"
#include "TextDecorationPainter.h"
#include "TextRun.h"

namespace WebCore {

class Color;
class Document;
class RenderCombineText;
class RenderStyle;
class RenderText;
class ShadowData;
struct CompositionUnderline;
struct MarkedText;
struct PaintInfo;
struct StyledMarkedText;

class TextBoxPainter {
public:
    TextBoxPainter(const LayoutIntegration::InlineContent&, const InlineDisplay::Box&, const RenderStyle&, PaintInfo&, const LayoutPoint& paintOffset);
    ~TextBoxPainter();

    void paint();

    static inline FloatSize rotateShadowOffset(const SpaceSeparatedPoint<Style::Length<>>& offset, WritingMode);

protected:
    auto& textBox() const { return m_textBox; }
    InlineIterator::TextBoxIterator makeIterator() const;

    void paintBackground();
    void paintForegroundAndDecorations();
    void paintCompositionBackground();
    void paintCompositionUnderlines();
    void paintCompositionForeground(const StyledMarkedText&);
    void paintPlatformDocumentMarkers();

    enum class BackgroundStyle { Normal, Rounded };
    void paintBackground(unsigned startOffset, unsigned endOffset, const Color&, BackgroundStyle = BackgroundStyle::Normal);
    void paintBackground(const StyledMarkedText&);
    void paintForeground(const StyledMarkedText&);
    TextDecorationPainter createDecorationPainter(const StyledMarkedText&, const FloatRect&);
    void paintBackgroundDecorations(TextDecorationPainter&, const StyledMarkedText&, const FloatRect&);
    void paintForegroundDecorations(TextDecorationPainter&, const StyledMarkedText&, const FloatRect&);
    void paintCompositionUnderline(const CompositionUnderline&, const FloatRoundedRect::Radii&, bool hasLiveConversion);
    void fillCompositionUnderline(float start, float width, const CompositionUnderline&, const FloatRoundedRect::Radii&, bool hasLiveConversion) const;
    void paintPlatformDocumentMarker(const MarkedText&);

    float textPosition();
    FloatRect computePaintRect(const LayoutPoint& paintOffset);
    bool computeHaveSelection() const;
    std::pair<unsigned, unsigned> selectionStartEnd() const;
    MarkedText createMarkedTextFromSelectionInBox();
    const FontCascade& fontCascade() const;
    WritingMode writingMode() const { return m_style.writingMode(); }
    FloatPoint textOriginFromPaintRect(const FloatRect&) const;

    struct DecoratingBox {
        InlineIterator::InlineBoxIterator inlineBox;
        const RenderStyle& style;
        TextDecorationPainter::Styles textDecorationStyles;
        FloatPoint location;
    };
    using DecoratingBoxList = Vector<DecoratingBox>;
    void collectDecoratingBoxesForBackgroundPainting(DecoratingBoxList&, const InlineIterator::TextBoxIterator&, FloatPoint textBoxLocation, const TextDecorationPainter::Styles&);

    // FIXME: We could just talk to the display box directly.
    const InlineIterator::BoxModernPath m_textBox;
    const RenderText& m_renderer;
    const Document& m_document;
    const RenderStyle& m_style;
    const FloatRect m_logicalRect;
    const TextRun m_paintTextRun;
    PaintInfo& m_paintInfo;
    const TextBoxSelectableRange m_selectableRange;
    const LayoutPoint m_paintOffset;
    const FloatRect m_paintRect;
    const bool m_isFirstLine;
    const bool m_isCombinedText;
    const bool m_isPrinting;
    const bool m_haveSelection;
    const bool m_containsComposition;
    const bool m_useCustomUnderlines;
    std::optional<bool> m_emphasisMarkExistsAndIsAbove { };
};

inline FloatSize TextBoxPainter::rotateShadowOffset(const SpaceSeparatedPoint<Style::Length<>>& offset, WritingMode writingMode)
{
    if (writingMode.isHorizontal())
        return { offset.x().value, offset.y().value };
    if (writingMode.isLineOverLeft()) // sideways-lr
        return { -offset.y().value, offset.x().value };
    return { offset.y().value, -offset.x().value };
}
}
