/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#include "BlockStepSizing.h"

#include "RenderBox.h"
#include "RenderStyleInlines.h"

namespace WebCore {

namespace BlockStepSizing {

bool childHasSupportedStyle(const RenderStyle& childStyle)
{
    return childStyle.blockStepInsert() == BlockStepInsert::MarginBox
        && childStyle.blockStepAlign() == BlockStepAlign::Auto
        && childStyle.blockStepRound() == BlockStepRound::Up;
}

LayoutUnit computeExtraSpace(LayoutUnit stepSize, LayoutUnit boxOuterSize)
{
    if (!stepSize)
        return { };

    if (!boxOuterSize)
        return stepSize;

    if (auto remainder = intMod(boxOuterSize, stepSize))
        return stepSize - remainder;
    return { };
}

void distributeExtraSpaceToChildMargins(RenderBox& child, LayoutUnit extraSpace, WritingMode containingBlockWritingMode)
{
    auto halfExtraSpace = extraSpace / 2;
    child.setMarginBefore(child.marginBefore(containingBlockWritingMode) + halfExtraSpace);
    child.setMarginAfter(child.marginAfter(containingBlockWritingMode) + halfExtraSpace);
}

NO_RETURN_DUE_TO_ASSERT void distributeExtraSpaceToChildPadding(RenderBox& /* child */, LayoutUnit /* extraSpace */, WritingMode /* containingBlockWritingMode */)
{
    ASSERT_NOT_IMPLEMENTED_YET();
}

NO_RETURN_DUE_TO_ASSERT void distributeExtraSpaceToChildContentArea(RenderBox& /* child */, LayoutUnit /* extraSpace */, WritingMode /* containingBlockWritingMode */)
{
    ASSERT_NOT_IMPLEMENTED_YET();
}

} // namespace BlockStepSizing

} // namespace WebCore
