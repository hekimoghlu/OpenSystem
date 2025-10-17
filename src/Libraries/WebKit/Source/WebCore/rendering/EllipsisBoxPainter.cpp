/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
#include "EllipsisBoxPainter.h"

#include "InlineIteratorTextBox.h"
#include "LineSelection.h"
#include "PaintInfo.h"
#include "RenderView.h"

namespace WebCore {

EllipsisBoxPainter::EllipsisBoxPainter(const InlineIterator::LineBox& lineBox, PaintInfo& paintInfo, const LayoutPoint& paintOffset, Color selectionForegroundColor, Color selectionBackgroundColor)
    : m_lineBox(lineBox)
    , m_paintInfo(paintInfo)
    , m_paintOffset(paintOffset)
    , m_selectionForegroundColor(selectionForegroundColor)
    , m_selectionBackgroundColor(selectionBackgroundColor)
{
}

void EllipsisBoxPainter::paint()
{
    // FIXME: Transition it to TextPainter.
    auto& context = m_paintInfo.context();
    auto& style = m_lineBox.style();
    auto textColor = style.visitedDependentColorWithColorFilter(CSSPropertyWebkitTextFillColor);

    if (m_paintInfo.forceTextColor())
        textColor = m_paintInfo.forcedTextColor();

    if (m_lineBox.ellipsisSelectionState() != RenderObject::HighlightState::None) {
        paintSelection();

        // Select the correct color for painting the text.
        auto foreground = m_paintInfo.forceTextColor() ? m_paintInfo.forcedTextColor() : m_selectionForegroundColor;
        if (foreground.isValid() && foreground != textColor)
            context.setFillColor(foreground);
    }

    if (textColor != context.fillColor())
        context.setFillColor(textColor);

    auto setShadow = false;
    if (style.textShadow()) {
        auto shadowColor = style.colorWithColorFilter(style.textShadow()->color());
        context.setDropShadow({ LayoutSize(style.textShadow()->x().value, style.textShadow()->y().value), style.textShadow()->radius().value, shadowColor, ShadowRadiusMode::Default });
        setShadow = true;
    }
    
    auto visualRect = m_lineBox.ellipsisVisualRect();
    auto textOrigin = visualRect.location();
    textOrigin.move(m_paintOffset.x(), m_paintOffset.y() + style.metricsOfPrimaryFont().intAscent());
    context.drawBidiText(style.fontCascade(), m_lineBox.ellipsisText(), textOrigin);

    if (textColor != context.fillColor())
        context.setFillColor(textColor);

    if (setShadow)
        context.clearDropShadow();
}

void EllipsisBoxPainter::paintSelection()
{
    auto& context = m_paintInfo.context();
    auto& style = m_lineBox.style();

    auto textColor = style.visitedDependentColorWithColorFilter(CSSPropertyColor);
    auto backgroundColor = m_selectionBackgroundColor;
    if (!backgroundColor.isVisible())
        return;

    // If the text color ends up being the same as the selection background, invert the selection background.
    if (textColor == backgroundColor)
        backgroundColor = backgroundColor.invertedColorWithAlpha(1.0);

    auto stateSaver = GraphicsContextStateSaver { context };

    auto visualRect = LayoutRect { m_lineBox.ellipsisVisualRect(InlineIterator::LineBox::AdjustedForSelection::Yes) };
    visualRect.move(m_paintOffset.x(), m_paintOffset.y());

    auto ellipsisText = m_lineBox.ellipsisText();
    constexpr bool canUseSimplifiedTextMeasuring = false;
    style.fontCascade().adjustSelectionRectForText(canUseSimplifiedTextMeasuring, ellipsisText, visualRect);
    context.fillRect(snapRectToDevicePixelsWithWritingDirection(visualRect, m_lineBox.formattingContextRoot().document().deviceScaleFactor(), ellipsisText.ltr()), backgroundColor);
}

}
