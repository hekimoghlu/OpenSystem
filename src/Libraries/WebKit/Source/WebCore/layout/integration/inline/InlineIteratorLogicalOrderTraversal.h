/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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

#include "InlineIteratorLineBox.h"
#include "InlineIteratorTextBox.h"
#include "RenderBlockFlow.h"

namespace WebCore {
namespace InlineIterator {

struct TextLogicalOrderCacheData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    Vector<TextBoxIterator> boxes;
    size_t index { 0 };
};
using TextLogicalOrderCache = std::unique_ptr<TextLogicalOrderCacheData>;

std::pair<TextBoxIterator, TextLogicalOrderCache> firstTextBoxInLogicalOrderFor(const RenderText&);
TextBoxIterator nextTextBoxInLogicalOrder(const TextBoxIterator&, TextLogicalOrderCache&);

struct LineLogicalOrderCacheData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    LineBoxIterator lineBox;
    Vector<LeafBoxIterator> boxes;
    size_t index { 0 };
};
using LineLogicalOrderCache = std::unique_ptr<LineLogicalOrderCacheData>;

LeafBoxIterator firstLeafOnLineInLogicalOrder(const LineBoxIterator&, LineLogicalOrderCache&);
LeafBoxIterator lastLeafOnLineInLogicalOrder(const LineBoxIterator&, LineLogicalOrderCache&);
LeafBoxIterator nextLeafOnLineInLogicalOrder(const LeafBoxIterator&, LineLogicalOrderCache&);
LeafBoxIterator previousLeafOnLineInLogicalOrder(const LeafBoxIterator&, LineLogicalOrderCache&);

LeafBoxIterator firstLeafOnLineInLogicalOrderWithNode(const LineBoxIterator&, LineLogicalOrderCache&);
LeafBoxIterator lastLeafOnLineInLogicalOrderWithNode(const LineBoxIterator&, LineLogicalOrderCache&);

template<typename ReverseFunction>
Vector<LeafBoxIterator> leafBoxesInLogicalOrder(const LineBoxIterator& lineBox, ReverseFunction&& reverseFunction)
{
    Vector<LeafBoxIterator> boxes;

    unsigned char minLevel = 128;
    unsigned char maxLevel = 0;

    for (auto box = lineBox->lineLeftmostLeafBox(); box; box = box.traverseLineRightwardOnLine()) {
        minLevel = std::min(minLevel, box->bidiLevel());
        maxLevel = std::max(maxLevel, box->bidiLevel());
        boxes.append(box);
    }

    if (lineBox->formattingContextRoot().style().rtlOrdering() == Order::Visual)
        return boxes;

    // Reverse of reordering of the line (L2 according to Bidi spec):
    // L2. From the highest level found in the text to the lowest odd level on each line,
    // reverse any contiguous sequence of characters that are at that level or higher.

    // Reversing the reordering of the line is only done up to the lowest odd level.
    if (!(minLevel % 2))
        ++minLevel;

    auto boxCount = boxes.size();
    for (; minLevel <= maxLevel; ++minLevel) {
        size_t boxIndex = 0;
        while (boxIndex < boxCount) {
            while (boxIndex < boxCount && boxes[boxIndex]->bidiLevel() < minLevel)
                ++boxIndex;

            auto first = boxIndex;
            while (boxIndex < boxCount && boxes[boxIndex]->bidiLevel() >= minLevel)
                ++boxIndex;

            reverseFunction(boxes.mutableSpan().subspan(first, boxIndex - first));
        }
    }

    return boxes;
}

}
}
