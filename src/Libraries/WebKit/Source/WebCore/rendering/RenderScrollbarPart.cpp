/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
#include "RenderScrollbarPart.h"

#include "PaintInfo.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderScrollbar.h"
#include "RenderScrollbarTheme.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderScrollbarPart);

RenderScrollbarPart::RenderScrollbarPart(Document& document, RenderStyle&& style, RenderScrollbar* scrollbar, ScrollbarPart part)
    : RenderBlock(Type::ScrollbarPart, document, WTFMove(style), { })
    , m_scrollbar(scrollbar)
    , m_part(part)
{
    ASSERT(isRenderScrollbarPart());
}

RenderScrollbarPart::~RenderScrollbarPart() = default;

void RenderScrollbarPart::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    setLocation(LayoutPoint()); // We don't worry about positioning ourselves. We're just determining our minimum width/height.
    if (m_scrollbar->orientation() == ScrollbarOrientation::Horizontal)
        layoutHorizontalPart();
    else
        layoutVerticalPart();

    clearNeedsLayout();
}

void RenderScrollbarPart::layoutHorizontalPart()
{
    if (m_part == ScrollbarBGPart) {
        setWidth(m_scrollbar->width());
        computeScrollbarHeight();
    } else {
        computeScrollbarWidth();
        setHeight(m_scrollbar->height());
    }
}

void RenderScrollbarPart::layoutVerticalPart()
{
    if (m_part == ScrollbarBGPart) {
        computeScrollbarWidth();
        setHeight(m_scrollbar->height());
    } else {
        setWidth(m_scrollbar->width());
        computeScrollbarHeight();
    } 
}

static int calcScrollbarThicknessUsing(RenderBox::SizeType sizeType, const Length& length)
{
    if ((!length.isPercentOrCalculated() && !length.isIntrinsicOrAuto()) || (sizeType == RenderBox::SizeType::MinSize && length.isAuto()))
        return minimumValueForLength(length, { });
    return ScrollbarTheme::theme().scrollbarThickness();
}

void RenderScrollbarPart::computeScrollbarWidth()
{
    if (!m_scrollbar->owningRenderer())
        return;
    int width = calcScrollbarThicknessUsing(RenderBox::SizeType::MainOrPreferredSize, style().width());
    int minWidth = calcScrollbarThicknessUsing(RenderBox::SizeType::MinSize, style().minWidth());
    int maxWidth = style().maxWidth().isUndefined() ? width : calcScrollbarThicknessUsing(RenderBox::SizeType::MaxSize, style().maxWidth());
    setWidth(std::max(minWidth, std::min(maxWidth, width)));
    
    // Buttons and track pieces can all have margins along the axis of the scrollbar. 
    m_marginBox.setLeft(minimumValueForLength(style().marginLeft(), { }));
    m_marginBox.setRight(minimumValueForLength(style().marginRight(), { }));
}

void RenderScrollbarPart::computeScrollbarHeight()
{
    if (!m_scrollbar->owningRenderer())
        return;
    int height = calcScrollbarThicknessUsing(RenderBox::SizeType::MainOrPreferredSize, style().height());
    int minHeight = calcScrollbarThicknessUsing(RenderBox::SizeType::MinSize, style().minHeight());
    int maxHeight = style().maxHeight().isUndefined() ? height : calcScrollbarThicknessUsing(RenderBox::SizeType::MaxSize, style().maxHeight());
    setHeight(std::max(minHeight, std::min(maxHeight, height)));

    // Buttons and track pieces can all have margins along the axis of the scrollbar. 
    m_marginBox.setTop(minimumValueForLength(style().marginTop(), { }));
    m_marginBox.setBottom(minimumValueForLength(style().marginBottom(), { }));
}

void RenderScrollbarPart::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    RenderBlock::styleDidChange(diff, oldStyle);
    setInline(false);
    clearPositionedState();
    setFloating(false);
    setHasNonVisibleOverflow(false);
    if (oldStyle && m_scrollbar && m_part != NoPart && diff >= StyleDifference::Repaint)
        m_scrollbar->theme().invalidatePart(*m_scrollbar, m_part);
}

void RenderScrollbarPart::imageChanged(WrappedImagePtr image, const IntRect* rect)
{
    if (m_scrollbar && m_part != NoPart)
        m_scrollbar->theme().invalidatePart(*m_scrollbar, m_part);
    else {
        if (view().frameView().isFrameViewScrollCorner(*this)) {
            view().frameView().invalidateScrollCorner(view().frameView().scrollCornerRect());
            return;
        }
        
        RenderBlock::imageChanged(image, rect);
    }
}

void RenderScrollbarPart::paintIntoRect(GraphicsContext& graphicsContext, const LayoutPoint& paintOffset, const LayoutRect& rect)
{
    // Make sure our dimensions match the rect.
    setLocation(rect.location() - toLayoutSize(paintOffset));
    setWidth(rect.width());
    setHeight(rect.height());

    if (graphicsContext.paintingDisabled() || !style().opacity())
        return;

    // We don't use RenderLayers for scrollbar parts, so we need to handle opacity here.
    // Opacity for ScrollbarBGPart is handled by RenderScrollbarTheme::willPaintScrollbar().
    bool needsTransparencyLayer = m_part != ScrollbarBGPart && style().opacity() < 1;
    if (needsTransparencyLayer) {
        graphicsContext.save();
        graphicsContext.clip(rect);
        graphicsContext.beginTransparencyLayer(style().opacity());
    }
    
    // Now do the paint.
    PaintInfo paintInfo(graphicsContext, snappedIntRect(rect), PaintPhase::BlockBackground, PaintBehavior::Normal);
    paint(paintInfo, paintOffset);
    paintInfo.phase = PaintPhase::ChildBlockBackgrounds;
    paint(paintInfo, paintOffset);
    paintInfo.phase = PaintPhase::Float;
    paint(paintInfo, paintOffset);
    paintInfo.phase = PaintPhase::Foreground;
    paint(paintInfo, paintOffset);
    paintInfo.phase = PaintPhase::Outline;
    paint(paintInfo, paintOffset);

    if (needsTransparencyLayer) {
        graphicsContext.endTransparencyLayer();
        graphicsContext.restore();
    }
}

RenderBox* RenderScrollbarPart::rendererOwningScrollbar() const
{
    if (!m_scrollbar)
        return nullptr;
    return m_scrollbar->owningRenderer();
}

}
