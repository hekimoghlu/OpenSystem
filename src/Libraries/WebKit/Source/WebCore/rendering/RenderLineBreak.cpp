/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#include "RenderLineBreak.h"

#include "Document.h"
#include "FontMetrics.h"
#include "HTMLElement.h"
#include "HTMLWBRElement.h"
#include "InlineIteratorBoxInlines.h"
#include "InlineIteratorLineBox.h"
#include "InlineIteratorSVGTextBox.h"
#include "InlineRunAndOffset.h"
#include "LineSelection.h"
#include "LogicalSelectionOffsetCaches.h"
#include "RenderBlock.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderView.h"
#include "SVGElementTypeHelpers.h"
#include "SVGInlineTextBox.h"
#include "VisiblePosition.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY)
#include "SelectionGeometry.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderLineBreak);

RenderLineBreak::RenderLineBreak(HTMLElement& element, RenderStyle&& style)
    : RenderBoxModelObject(Type::LineBreak, element, WTFMove(style), { }, is<HTMLWBRElement>(element) ? OptionSet<LineBreakFlag> { LineBreakFlag::IsWBR } : OptionSet<LineBreakFlag> { })
{
    ASSERT(isRenderLineBreak());
}

RenderLineBreak::~RenderLineBreak()
{
}

LayoutUnit RenderLineBreak::lineHeight(bool firstLine, LineDirectionMode /*direction*/, LinePositionMode /*linePositionMode*/) const
{
    if (firstLine) {
        auto& firstLineStyle = this->firstLineStyle();
        if (&firstLineStyle != &style())
            return LayoutUnit::fromFloatCeil(firstLineStyle.computedLineHeight());
    }

    if (!m_cachedLineHeight)
        m_cachedLineHeight = LayoutUnit::fromFloatCeil(style().computedLineHeight());
    return *m_cachedLineHeight;
}

LayoutUnit RenderLineBreak::baselinePosition(FontBaseline baselineType, bool firstLine, LineDirectionMode direction, LinePositionMode linePositionMode) const
{
    auto& style = firstLine ? firstLineStyle() : this->style();
    auto& fontMetrics = style.metricsOfPrimaryFont();
    return LayoutUnit { (fontMetrics.ascent(baselineType) + (lineHeight(firstLine, direction, linePositionMode) - fontMetrics.height()) / 2) };
}

int RenderLineBreak::caretMinOffset() const
{
    return 0;
}

int RenderLineBreak::caretMaxOffset() const
{ 
    return 1;
}

bool RenderLineBreak::canBeSelectionLeaf() const
{
    return true;
}

VisiblePosition RenderLineBreak::positionForPoint(const LayoutPoint&, HitTestSource, const RenderFragmentContainer*)
{
    return createVisiblePosition(0, Affinity::Downstream);
}

IntRect RenderLineBreak::linesBoundingBox() const
{
    auto run = InlineIterator::boxFor(*this);
    if (!run)
        return { };

    return enclosingIntRect(run->visualRectIgnoringBlockDirection());
}

void RenderLineBreak::boundingRects(Vector<LayoutRect>& rects, const LayoutPoint& accumulatedOffset) const
{
    auto box = InlineIterator::boxFor(*this);
    if (!box)
        return;

    auto rect = LayoutRect { box->visualRectIgnoringBlockDirection() };
    rect.moveBy(accumulatedOffset);
    rects.append(rect);
}

void RenderLineBreak::absoluteQuads(Vector<FloatQuad>& quads, bool* wasFixed) const
{
    auto box = InlineIterator::boxFor(*this);
    if (!box)
        return;

    auto rect = box->visualRectIgnoringBlockDirection();
    quads.append(localToAbsoluteQuad(FloatRect(rect.location(), rect.size()), UseTransforms, wasFixed));
}

void RenderLineBreak::updateFromStyle()
{
    m_cachedLineHeight = { };
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(isInline());
}

#if PLATFORM(IOS_FAMILY)
void RenderLineBreak::collectSelectionGeometries(Vector<SelectionGeometry>& rects, unsigned, unsigned)
{
    auto run = InlineIterator::boxFor(*this);

    if (!run)
        return;
    auto lineBox = run->lineBox();

    auto lineSelectionRect = LineSelection::logicalRect(*lineBox);
    LayoutRect rect = IntRect(run->logicalLeftIgnoringInlineDirection(), lineSelectionRect.y(), 0, lineSelectionRect.height());
    if (!lineBox->isHorizontal())
        rect = rect.transposedRect();

    if (lineBox->isFirstAfterPageBreak()) {
        if (run->isHorizontal())
            rect.shiftYEdgeTo(lineBox->logicalTop());
        else
            rect.shiftXEdgeTo(lineBox->logicalTop());
    }

    // FIXME: Out-of-flow positioned line breaks do not follow normal containing block chain.
    auto* containingBlock = RenderObject::containingBlockForPositionType(PositionType::Static, *this);
    // Map rect, extended left to leftOffset, and right to rightOffset, through transforms to get minX and maxX.
    LogicalSelectionOffsetCaches cache(*containingBlock);
    LayoutUnit leftOffset = containingBlock->logicalLeftSelectionOffset(*containingBlock, LayoutUnit(run->logicalTop()), cache);
    LayoutUnit rightOffset = containingBlock->logicalRightSelectionOffset(*containingBlock, LayoutUnit(run->logicalTop()), cache);
    LayoutRect extentsRect = rect;
    if (run->isHorizontal()) {
        extentsRect.setX(leftOffset);
        extentsRect.setWidth(rightOffset - leftOffset);
    } else {
        extentsRect.setY(leftOffset);
        extentsRect.setHeight(rightOffset - leftOffset);
    }
    extentsRect = localToAbsoluteQuad(FloatRect(extentsRect)).enclosingBoundingBox();
    if (!run->isHorizontal())
        extentsRect = extentsRect.transposedRect();
    bool isFirstOnLine = !run->nextLineLeftwardOnLine();
    bool isLastOnLine = !run->nextLineRightwardOnLine();

    bool isFixed = false;
    auto absoluteQuad = localToAbsoluteQuad(FloatRect(rect), UseTransforms, &isFixed);
    bool boxIsHorizontal = !is<InlineIterator::SVGTextBoxIterator>(run) ? run->isHorizontal() : !writingMode().isVertical();

    rects.append(SelectionGeometry(absoluteQuad, HTMLElement::selectionRenderingBehavior(element()), run->direction(), extentsRect.x(), extentsRect.maxX(), extentsRect.maxY(), 0, run->isLineBreak(), isFirstOnLine, isLastOnLine, false, false, boxIsHorizontal, isFixed, view().pageNumberForBlockProgressionOffset(absoluteQuad.enclosingBoundingBox().x())));
}
#endif

} // namespace WebCore
