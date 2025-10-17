/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#include "RenderBoxModelObjectInlines.h"

namespace WebCore {
namespace InlineIterator {

inline float previousLineBoxContentBottomOrBorderAndPadding(const LineBox& lineBox)
{
    return lineBox.isFirst() ? lineBox.formattingContextRoot().borderAndPaddingBefore().toFloat() : lineBox.contentLogicalTopAdjustedForPrecedingLineBox();
}

inline float contentStartInBlockDirection(const LineBox& lineBox)
{
    if (!lineBox.formattingContextRoot().writingMode().isBlockFlipped())
        return std::max(lineBox.contentLogicalTop(), previousLineBoxContentBottomOrBorderAndPadding(lineBox));
    return std::min(lineBox.contentLogicalBottom(), lineBox.contentLogicalBottomAdjustedForFollowingLineBox());
}

inline LeafBoxIterator LineBox::logicalLeftmostLeafBox() const
{
    return formattingContextRoot().writingMode().isLogicalLeftLineLeft()
        ? lineLeftmostLeafBox()
        : lineRightmostLeafBox();
}

inline LeafBoxIterator LineBox::logicalRightmostLeafBox() const
{
    return formattingContextRoot().writingMode().isLogicalLeftLineLeft()
        ? lineRightmostLeafBox()
        : lineLeftmostLeafBox();
}

}
}
