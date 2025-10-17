/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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

#include "RenderBlock.h"
#include "RenderStyleInlines.h"

namespace WebCore {

inline LayoutUnit RenderBlock::endOffsetForContent() const { return !writingMode().isLogicalLeftInlineStart() ? logicalLeftOffsetForContent() : logicalWidth() - logicalRightOffsetForContent(); }
inline LayoutUnit RenderBlock::logicalMarginBoxHeightForChild(const RenderBox& child) const { return isHorizontalWritingMode() ? child.marginBoxRect().height() : child.marginBoxRect().width(); }
inline LayoutUnit RenderBlock::startOffsetForContent() const { return writingMode().isLogicalLeftInlineStart() ? logicalLeftOffsetForContent() : logicalWidth() - logicalRightOffsetForContent(); }
inline LayoutUnit RenderBlock::logicalRightOffsetForLine(LayoutUnit position, LayoutUnit logicalHeight) const { return adjustLogicalRightOffsetForLine(logicalRightFloatOffsetForLine(position, logicalRightOffsetForContent(), logicalHeight)); }
inline LayoutUnit RenderBlock::logicalLeftOffsetForLine(LayoutUnit position, LayoutUnit logicalHeight) const { return adjustLogicalLeftOffsetForLine(logicalLeftFloatOffsetForLine(position, logicalLeftOffsetForContent(), logicalHeight)); }

inline LayoutUnit RenderBlock::endOffsetForLine(LayoutUnit position, LayoutUnit logicalHeight) const
{
    return !writingMode().isLogicalLeftInlineStart() ? logicalLeftOffsetForLine(position, logicalHeight) : logicalWidth() - logicalRightOffsetForLine(position, logicalHeight);
}

inline bool RenderBlock::shouldSkipCreatingRunsForObject(RenderObject& object)
{
    return object.isFloating() || (object.isOutOfFlowPositioned() && !object.style().isOriginalDisplayInlineType() && !object.container()->isRenderInline());
}

inline LayoutUnit RenderBlock::startOffsetForLine(LayoutUnit position, LayoutUnit logicalHeight) const
{
    return writingMode().isLogicalLeftInlineStart() ? logicalLeftOffsetForLine(position, logicalHeight)
        : logicalWidth() - logicalRightOffsetForLine(position, logicalHeight);
}

// Versions that can compute line offsets with the fragment and page offset passed in. Used for speed to avoid having to
// compute the fragment all over again when you already know it.
inline LayoutUnit RenderBlock::availableLogicalWidthForLine(LayoutUnit position, LayoutUnit logicalHeight) const
{
    auto logicalRightOffsetForLine = adjustLogicalRightOffsetForLine(logicalRightFloatOffsetForLine(position, logicalRightOffsetForContent(), logicalHeight));
    auto logicalLeftOffsetForLine = adjustLogicalLeftOffsetForLine(logicalLeftFloatOffsetForLine(position, logicalLeftOffsetForContent(), logicalHeight));
    return std::max(0_lu, logicalRightOffsetForLine - logicalLeftOffsetForLine);
}

} // namespace WebCore
