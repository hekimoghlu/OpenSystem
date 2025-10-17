/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

#include "GridPositionsResolver.h"
#include "LayoutUnit.h"
#include "RenderBox.h"

namespace WebCore {

enum class ItemPosition : uint8_t;
class RenderElement;
class RenderGrid;

enum class GridAxis : uint8_t {
    GridRowAxis = 1 << 0,
    GridColumnAxis = 1 << 1
};

struct ExtraMarginsFromSubgrids {
    inline LayoutUnit extraTrackStartMargin() const { return m_extraMargins.first; }
    inline LayoutUnit extraTrackEndMargin() const { return m_extraMargins.second; }
    inline LayoutUnit extraTotalMargin() const { return m_extraMargins.first + m_extraMargins.second; }

    ExtraMarginsFromSubgrids& operator+=(const ExtraMarginsFromSubgrids& rhs)
    {
        m_extraMargins.first += rhs.extraTrackStartMargin();
        m_extraMargins.second += rhs.extraTrackEndMargin();
        return *this;
    }

    void addTrackStartMargin(LayoutUnit extraMargin) { m_extraMargins.first += extraMargin; }
    void addTrackEndMargin(LayoutUnit extraMargin) { m_extraMargins.second += extraMargin; }

    std::pair<LayoutUnit, LayoutUnit> m_extraMargins;
};

namespace GridLayoutFunctions {

LayoutUnit computeMarginLogicalSizeForGridItem(const RenderGrid&, GridTrackSizingDirection, const RenderBox&);
LayoutUnit marginLogicalSizeForGridItem(const RenderGrid&, GridTrackSizingDirection, const RenderBox&);
void setOverridingContentSizeForGridItem(const RenderGrid&, RenderBox& gridItem, LayoutUnit, GridTrackSizingDirection);
void clearOverridingContentSizeForGridItem(const RenderGrid&, RenderBox& gridItem, GridTrackSizingDirection);
bool isOrthogonalGridItem(const RenderGrid&, const RenderBox&);
bool isGridItemInlineSizeDependentOnBlockConstraints(const RenderBox& gridItem, const RenderGrid& parentGrid, ItemPosition gridItemAlignSelf);
bool isOrthogonalParent(const RenderGrid&, const RenderElement& parent);
bool isAspectRatioBlockSizeDependentGridItem(const RenderBox&);
GridTrackSizingDirection flowAwareDirectionForGridItem(const RenderGrid&, const RenderBox&, GridTrackSizingDirection);
GridTrackSizingDirection flowAwareDirectionForParent(const RenderGrid&, const RenderElement& parent, GridTrackSizingDirection);
std::optional<RenderBox::GridAreaSize> overridingContainingBlockContentSizeForGridItem(const RenderBox&, GridTrackSizingDirection);
bool hasRelativeOrIntrinsicSizeForGridItem(const RenderBox& gridItem, GridTrackSizingDirection);

bool isFlippedDirection(const RenderGrid&, GridTrackSizingDirection);
bool isSubgridReversedDirection(const RenderGrid&, GridTrackSizingDirection outerDirection, const RenderGrid& subgrid);
ExtraMarginsFromSubgrids extraMarginForSubgridAncestors(GridTrackSizingDirection, const RenderBox& gridItem);

unsigned alignmentContextForBaselineAlignment(const GridSpan&, const ItemPosition& alignment);

GridAxis gridAxisForDirection(GridTrackSizingDirection);
GridTrackSizingDirection gridDirectionForAxis(GridAxis);

}

} // namespace WebCore

