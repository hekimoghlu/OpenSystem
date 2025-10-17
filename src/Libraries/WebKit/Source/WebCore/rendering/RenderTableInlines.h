/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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
#include "RenderTable.h"

namespace WebCore {

inline LayoutUnit RenderTable::borderBottom() const
{
    if (writingMode().isHorizontal())
        return writingMode().isBlockTopToBottom() ? borderAfter() : borderBefore();
    return writingMode().isInlineTopToBottom() ? borderEnd() : borderStart();
}

inline LayoutUnit RenderTable::borderLeft() const
{
    if (writingMode().isHorizontal())
        return writingMode().isInlineLeftToRight() ? borderStart() : borderEnd();
    return writingMode().isBlockLeftToRight() ? borderBefore() : borderAfter();
}

inline LayoutUnit RenderTable::borderRight() const
{
    if (writingMode().isHorizontal())
        return writingMode().isInlineLeftToRight() ? borderEnd() : borderStart();
    return writingMode().isBlockLeftToRight() ? borderAfter() : borderBefore();
}

inline LayoutUnit RenderTable::borderTop() const
{
    if (writingMode().isHorizontal())
        return writingMode().isBlockTopToBottom() ? borderBefore() : borderAfter();
    return writingMode().isInlineTopToBottom() ? borderStart() : borderEnd();
}

inline RectEdges<LayoutUnit> RenderTable::borderWidths() const
{
    return {
        borderTop(),
        borderRight(),
        borderBottom(),
        borderLeft()
    };
}

inline LayoutUnit RenderTable::bordersPaddingAndSpacingInRowDirection() const
{
    // 'border-spacing' only applies to separate borders (see 17.6.1 The separated borders model).
    return borderStart() + borderEnd() + (collapseBorders() ? 0_lu : (paddingStart() + paddingEnd() + borderSpacingInRowDirection()));
}

inline LayoutUnit RenderTable::outerBorderBottom() const
{
    if (writingMode().isHorizontal())
        return writingMode().isBlockTopToBottom() ? outerBorderAfter() : outerBorderBefore();
    return writingMode().isInlineTopToBottom() ? outerBorderEnd() : outerBorderStart();
}

inline LayoutUnit RenderTable::outerBorderLeft() const
{
    if (writingMode().isHorizontal())
        return writingMode().isInlineLeftToRight() ? outerBorderStart() : outerBorderEnd();
    return writingMode().isBlockLeftToRight() ? outerBorderBefore() : outerBorderAfter();
}

inline LayoutUnit RenderTable::outerBorderRight() const
{
    if (writingMode().isHorizontal())
        return writingMode().isInlineLeftToRight() ? outerBorderEnd() : outerBorderStart();
    return writingMode().isBlockLeftToRight() ? outerBorderAfter() : outerBorderBefore();
}

inline LayoutUnit RenderTable::outerBorderTop() const
{
    if (writingMode().isHorizontal())
        return writingMode().isBlockTopToBottom() ? outerBorderBefore() : outerBorderAfter();
    return writingMode().isInlineTopToBottom() ? outerBorderStart() : borderEnd();
}

} // namespace WebCore
