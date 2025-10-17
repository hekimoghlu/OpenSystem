/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#include "LineWidth.h"

#include "RenderBlockFlow.h"
#include "RenderBlockInlines.h"
#include "RenderBoxInlines.h"
#include "RenderStyleInlines.h"

namespace WebCore {

LineWidth::LineWidth(RenderBlockFlow& block, bool isFirstLine)
    : m_block(block)
    , m_isFirstLine(isFirstLine)
{
    updateAvailableWidth();
}

bool LineWidth::fitsOnLine(bool ignoringTrailingSpace) const
{
    return ignoringTrailingSpace ? fitsOnLineExcludingTrailingCollapsedWhitespace() : fitsOnLineIncludingExtraWidth(0);
}

bool LineWidth::fitsOnLineIncludingExtraWidth(float extra) const
{
    auto adjustedCurrentWidth = currentWidth() + extra;
    return adjustedCurrentWidth < m_availableWidth || WTF::areEssentiallyEqual(adjustedCurrentWidth, m_availableWidth);
}

bool LineWidth::fitsOnLineExcludingTrailingWhitespace(float extra) const
{
    auto adjustedCurrentWidth = currentWidth() - m_trailingWhitespaceWidth + extra;
    return adjustedCurrentWidth < m_availableWidth || WTF::areEssentiallyEqual(adjustedCurrentWidth, m_availableWidth);
}

bool LineWidth::fitsOnLineExcludingTrailingCollapsedWhitespace() const
{
    auto adjustedCurrentWidth = currentWidth() - m_trailingCollapsedWhitespaceWidth;
    return adjustedCurrentWidth < m_availableWidth || WTF::areEssentiallyEqual(adjustedCurrentWidth, m_availableWidth);
}

void LineWidth::updateAvailableWidth(LayoutUnit replacedHeight)
{
    LayoutUnit height = m_block.logicalHeight();
    LayoutUnit logicalHeight = m_block.minLineHeightForReplacedRenderer(m_isFirstLine, replacedHeight);
    m_left = m_block.logicalLeftOffsetForLine(height, logicalHeight);
    m_right = m_block.logicalRightOffsetForLine(height, logicalHeight);

    computeAvailableWidthFromLeftAndRight();
}

static bool newFloatShrinksLine(const FloatingObject& newFloat, const RenderBlockFlow& block, bool isFirstLine)
{
    LayoutUnit blockOffset = block.logicalHeight();
    if (blockOffset >= block.logicalTopForFloat(newFloat) && blockOffset < block.logicalBottomForFloat(newFloat))
        return true;

    // initial-letter float always shrinks the first line.
    const auto& style = newFloat.renderer().style();
    if (isFirstLine && style.pseudoElementType() == PseudoId::FirstLetter && !style.initialLetter().isEmpty())
        return true;
    return false;
}

void LineWidth::shrinkAvailableWidthForNewFloatIfNeeded(const FloatingObject& newFloat)
{
    if (!newFloatShrinksLine(newFloat, m_block, m_isFirstLine))
        return;
    ShapeOutsideDeltas shapeDeltas;
    if (ShapeOutsideInfo* shapeOutsideInfo = newFloat.renderer().shapeOutsideInfo()) {
        LayoutUnit lineHeight = m_block.lineHeight(m_isFirstLine, m_block.isHorizontalWritingMode() ? HorizontalLine : VerticalLine, PositionOfInteriorLineBoxes);
        shapeDeltas = shapeOutsideInfo->computeDeltasForContainingBlockLine(m_block, newFloat, m_block.logicalHeight(), lineHeight);
    }

    if (newFloat.type() == FloatingObject::FloatLeft) {
        float newLeft = m_block.logicalRightForFloat(newFloat);
        if (shapeDeltas.isValid()) {
            if (shapeDeltas.lineOverlapsShape())
                newLeft += shapeDeltas.rightMarginBoxDelta();
            else // If the line doesn't overlap the shape, then we need to act as if this float didn't exist.
                newLeft = m_left;
        }
        m_left = std::max<float>(m_left, newLeft);
    } else {
        float newRight = m_block.logicalLeftForFloat(newFloat);
        if (shapeDeltas.isValid()) {
            if (shapeDeltas.lineOverlapsShape())
                newRight += shapeDeltas.leftMarginBoxDelta();
            else // If the line doesn't overlap the shape, then we need to act as if this float didn't exist.
                newRight = m_right;
        }
        m_right = std::min<float>(m_right, newRight);
    }

    computeAvailableWidthFromLeftAndRight();
}

void LineWidth::commit()
{
    m_committedWidth += m_uncommittedWidth;
    m_uncommittedWidth = 0;
    if (m_hasUncommittedReplaced) {
        m_hasCommittedReplaced = true;
        m_hasUncommittedReplaced = false;
    }
    m_hasCommitted = true;
}

inline static float availableWidthAtOffset(const RenderBlockFlow& block, const LayoutUnit& offset,
    float& newLineLeft, float& newLineRight, const LayoutUnit& lineHeight = 0)
{
    newLineLeft = block.logicalLeftOffsetForLine(offset, lineHeight);
    newLineRight = block.logicalRightOffsetForLine(offset, lineHeight);
    return std::max(0.0f, newLineRight - newLineLeft);
}

void LineWidth::updateLineDimension(LayoutUnit newLineTop, LayoutUnit newLineWidth, float newLineLeft, float newLineRight)
{
    if (newLineWidth <= m_availableWidth)
        return;

    m_block.setLogicalHeight(newLineTop);
    m_availableWidth = newLineWidth;
    m_left = newLineLeft;
    m_right = newLineRight;
}

void LineWidth::wrapNextToShapeOutside(bool isFirstLine)
{
    LayoutUnit lineHeight = m_block.lineHeight(isFirstLine, m_block.isHorizontalWritingMode() ? HorizontalLine : VerticalLine, PositionOfInteriorLineBoxes);
    LayoutUnit lineLogicalTop = m_block.logicalHeight();
    LayoutUnit newLineTop = lineLogicalTop;
    LayoutUnit floatLogicalBottom = m_block.nextFloatLogicalBottomBelow(lineLogicalTop);

    float newLineWidth;
    float newLineLeft = m_left;
    float newLineRight = m_right;
    while (true) {
        newLineWidth = availableWidthAtOffset(m_block, newLineTop, newLineLeft, newLineRight, lineHeight);
        if (newLineWidth >= m_uncommittedWidth)
            break;

        if (newLineTop >= floatLogicalBottom)
            break;

        ++newLineTop;
    }
    updateLineDimension(newLineTop, LayoutUnit(newLineWidth), LayoutUnit(newLineLeft), LayoutUnit(newLineRight));
}

void LineWidth::fitBelowFloats(bool isFirstLine)
{
    ASSERT(!m_committedWidth);
    ASSERT(!fitsOnLine());

    LayoutUnit floatLogicalBottom;
    LayoutUnit lastFloatLogicalBottom = m_block.logicalHeight();
    float newLineWidth = m_availableWidth;
    float newLineLeft = m_left;
    float newLineRight = m_right;

    FloatingObject* lastFloatFromPreviousLine = (m_block.containsFloats() ? m_block.m_floatingObjects->set().last().get() : 0);
    if (lastFloatFromPreviousLine && lastFloatFromPreviousLine->renderer().shapeOutsideInfo())
        return wrapNextToShapeOutside(isFirstLine);

    while (true) {
        floatLogicalBottom = m_block.nextFloatLogicalBottomBelow(lastFloatLogicalBottom);
        if (floatLogicalBottom <= lastFloatLogicalBottom)
            break;

        newLineWidth = availableWidthAtOffset(m_block, floatLogicalBottom, newLineLeft, newLineRight);
        lastFloatLogicalBottom = floatLogicalBottom;

        if (newLineWidth >= m_uncommittedWidth)
            break;
    }

    updateLineDimension(lastFloatLogicalBottom, LayoutUnit(newLineWidth), LayoutUnit(newLineLeft), LayoutUnit(newLineRight));
}

void LineWidth::setTrailingWhitespaceWidth(float collapsedWhitespace, float borderPaddingMargin)
{
    m_trailingCollapsedWhitespaceWidth = collapsedWhitespace;
    m_trailingWhitespaceWidth = collapsedWhitespace + borderPaddingMargin;
}

void LineWidth::computeAvailableWidthFromLeftAndRight()
{
    m_availableWidth = std::max<float>(0, m_right - m_left);
}

}
