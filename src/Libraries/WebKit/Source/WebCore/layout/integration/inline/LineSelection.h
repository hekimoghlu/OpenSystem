/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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

#include "InlineIteratorBox.h"
#include "InlineIteratorLineBox.h"
#include "RenderBlockFlow.h"

namespace WebCore {

class LineSelection {
public:
    static float logicalTop(const InlineIterator::LineBox& lineBox) { return lineBox.contentLogicalTopAdjustedForPrecedingLineBox(); }
    static float logicalBottom(const InlineIterator::LineBox& lineBox) { return lineBox.contentLogicalBottomAdjustedForFollowingLineBox(); }

    static FloatRect logicalRect(const InlineIterator::LineBox& lineBox)
    {
        return { FloatPoint { lineBox.contentLogicalLeft(), lineBox.contentLogicalTopAdjustedForPrecedingLineBox() }, FloatPoint { lineBox.contentLogicalRight(), lineBox.contentLogicalBottomAdjustedForFollowingLineBox() } };
    }

    static FloatRect physicalRect(const InlineIterator::LineBox& lineBox)
    {
        auto physicalRect = logicalRect(lineBox);
        if (!lineBox.isHorizontal())
            physicalRect = physicalRect.transposedRect();
        lineBox.formattingContextRoot().flipForWritingMode(physicalRect);
        return physicalRect;
    }

    static float logicalTopAdjustedForPrecedingBlock(const InlineIterator::LineBox& lineBox)
    {
        // FIXME: Move adjustEnclosingTopForPrecedingBlock from RenderBlockFlow to here.
        return lineBox.formattingContextRoot().adjustEnclosingTopForPrecedingBlock(LayoutUnit { lineBox.contentLogicalTopAdjustedForPrecedingLineBox() });
    }

    static RenderObject::HighlightState selectionState(const InlineIterator::LineBox& lineBox)
    {
        auto& root = lineBox.formattingContextRoot();
        if (root.selectionState() == RenderObject::HighlightState::None)
            return RenderObject::HighlightState::None;

        auto lineState = RenderObject::HighlightState::None;
        for (auto box = lineBox.lineLeftmostLeafBox(); box; box.traverseLineRightwardOnLine()) {
            auto boxState = box->selectionState();
            if (lineState == RenderObject::HighlightState::None)
                lineState = boxState;
            else if (lineState == RenderObject::HighlightState::Start) {
                if (boxState == RenderObject::HighlightState::End || boxState == RenderObject::HighlightState::None)
                    lineState = RenderObject::HighlightState::Both;
            } else if (lineState == RenderObject::HighlightState::Inside) {
                if (boxState == RenderObject::HighlightState::Start || boxState == RenderObject::HighlightState::End)
                    lineState = boxState;
                else if (boxState == RenderObject::HighlightState::None)
                    lineState = RenderObject::HighlightState::End;
            } else if (lineState == RenderObject::HighlightState::End) {
                if (boxState == RenderObject::HighlightState::Start)
                    lineState = RenderObject::HighlightState::Both;
            }

            if (lineState == RenderObject::HighlightState::Both)
                break;
        }
        return lineState;
    }

};

}

