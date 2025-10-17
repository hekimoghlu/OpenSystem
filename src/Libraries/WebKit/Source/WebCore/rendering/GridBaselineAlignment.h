/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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

#include "BaselineAlignment.h"
#include "GridLayoutFunctions.h"
#include "RenderStyleConstants.h"
#include "WritingMode.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>

namespace WebCore {

// This is the class that implements the Baseline Alignment logic, using internally the BaselineAlignmentState and
// BaselineGroup classes.
//
// The first phase is to collect the items that will participate in baseline alignment together. During this
// phase the required baseline-sharing groups will be created for each Baseline alignment-context shared by
// the items participating in the baseline alignment.
//
// Additionally, the baseline-sharing groups' offsets, max-ascend and max-descent will be computed and stored.
// This class also computes the baseline offset for a particular item, based on the max-ascent for its associated
// baseline-sharing group.
class GridBaselineAlignment {
public:
    // Collects the items participating in baseline alignment and updates the corresponding baseline-sharing
    // group of the Baseline Context the items belongs to.
    // All the baseline offsets are updated accordingly based on the added item.
    void updateBaselineAlignmentContext(ItemPosition, unsigned sharedContext, const RenderBox&, GridAxis);

    // Returns the baseline offset of a particular item, based on the max-ascent for its associated
    // baseline-sharing group
    LayoutUnit baselineOffsetForGridItem(ItemPosition, unsigned sharedContext, const RenderBox&, GridAxis) const;

    // Sets the Grid Container's writing mode so that we can avoid the dependecy of the LayoutGrid class for
    // determining whether a grid item is orthogonal or not.
    void setWritingMode(WritingMode writingMode) { m_writingMode = writingMode; };

    // Clearing the Baseline Alignment context and their internal classes and data structures.
    void clear(GridAxis);

private:
    const BaselineGroup& baselineGroupForGridItem(ItemPosition, unsigned sharedContext, const RenderBox&, GridAxis) const;
    LayoutUnit marginOverForGridItem(const RenderBox&, GridAxis) const;
    LayoutUnit marginUnderForGridItem(const RenderBox&, GridAxis) const;
    LayoutUnit logicalAscentForGridItem(const RenderBox&, GridAxis, ItemPosition) const;
    LayoutUnit ascentForGridItem(const RenderBox&, GridAxis, ItemPosition) const;
    LayoutUnit descentForGridItem(const RenderBox&, LayoutUnit, GridAxis, ExtraMarginsFromSubgrids) const;
    bool isDescentBaselineForGridItem(const RenderBox&, GridAxis) const;
    bool isVerticalAlignmentContext(GridAxis) const;
    bool isOrthogonalGridItemForBaseline(const RenderBox&) const;
    bool isParallelToAlignmentAxisForGridItem(const RenderBox&, GridAxis) const;

    typedef UncheckedKeyHashMap<unsigned, std::unique_ptr<BaselineAlignmentState>, DefaultHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>> BaselineAlignmentStateMap;

    // Grid Container's writing mode, used to determine grid item's orthogonality.
    WritingMode m_writingMode;
    BaselineAlignmentStateMap m_rowAxisBaselineAlignmentStates;
    BaselineAlignmentStateMap m_colAxisBaselineAlignmentStates;
};

} // namespace WebCore
