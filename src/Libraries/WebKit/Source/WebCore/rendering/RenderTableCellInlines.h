/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#include "RenderStyleInlines.h"
#include "RenderTableCell.h"
#include "StyleContentAlignmentData.h"

namespace WebCore {

inline const BorderValue& RenderTableCell::borderAdjoiningCellAfter(const RenderTableCell& cell)
{
    ASSERT_UNUSED(cell, table()->cellBefore(&cell) == this);
    return style().borderEnd(tableWritingMode());
}

inline const BorderValue& RenderTableCell::borderAdjoiningCellBefore(const RenderTableCell& cell)
{
    ASSERT_UNUSED(cell, table()->cellAfter(&cell) == this);
    return style().borderStart(tableWritingMode());
}

inline const BorderValue& RenderTableCell::borderAdjoiningTableEnd() const
{
    ASSERT(isFirstOrLastCellInRow());
    return style().borderEnd(tableWritingMode());
}

inline const BorderValue& RenderTableCell::borderAdjoiningTableStart() const
{
    ASSERT(isFirstOrLastCellInRow());
    return style().borderStart(tableWritingMode());
}

inline LayoutUnit RenderTableCell::logicalHeightForRowSizing() const
{
    // FIXME: This function does too much work, and is very hot during table layout!
    LayoutUnit adjustedLogicalHeight = logicalHeight() - (intrinsicPaddingBefore() + intrinsicPaddingAfter());
    if (!style().logicalHeight().isSpecified())
        return adjustedLogicalHeight;
    LayoutUnit styleLogicalHeight = valueForLength(style().logicalHeight(), 0);
    // In strict mode, box-sizing: content-box do the right thing and actually add in the border and padding.
    // Call computedCSSPadding* directly to avoid including implicitPadding.
    if (!document().inQuirksMode() && style().boxSizing() != BoxSizing::BorderBox)
        styleLogicalHeight += computedCSSPaddingBefore() + computedCSSPaddingAfter() + borderBefore() + borderAfter();
    return std::max(styleLogicalHeight, adjustedLogicalHeight);
}

inline Length RenderTableCell::styleOrColLogicalWidth() const
{
    Length styleWidth = style().logicalWidth();
    if (!styleWidth.isAuto())
        return styleWidth;
    if (RenderTableCol* firstColumn = table()->colElement(col()))
        return logicalWidthFromColumns(firstColumn, styleWidth);
    return styleWidth;
}

inline bool RenderTableCell::isBaselineAligned() const
{
    if (auto alignContent = style().alignContent(); !alignContent.isNormal())
        return alignContent.position() == ContentPosition::Baseline;

    VerticalAlign va = style().verticalAlign();
    return va == VerticalAlign::Baseline || va == VerticalAlign::TextBottom || va == VerticalAlign::TextTop || va == VerticalAlign::Super || va == VerticalAlign::Sub || va == VerticalAlign::Length;
}

} // namespace WebCore
