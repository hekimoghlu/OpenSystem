/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#include "RenderGrid.h"

class RenderGrid;

namespace WebCore {

class AncestorSubgridIterator {
public:
    AncestorSubgridIterator();
    AncestorSubgridIterator(SingleThreadWeakPtr<RenderGrid> firstAncestorSubgrid, GridTrackSizingDirection);

    RenderGrid& operator*();

    bool operator==(const AncestorSubgridIterator&) const;

    AncestorSubgridIterator& operator++();
    AncestorSubgridIterator begin();
    AncestorSubgridIterator end();
private:
    AncestorSubgridIterator(SingleThreadWeakPtr<RenderGrid> firstAncestorSubgrid, SingleThreadWeakPtr<RenderGrid> currentAncestor, GridTrackSizingDirection);
    AncestorSubgridIterator(SingleThreadWeakPtr<RenderGrid> firstAncestorSubgrid, SingleThreadWeakPtr<RenderGrid> currentAncestor, std::optional<GridTrackSizingDirection>);

    const SingleThreadWeakPtr<const RenderGrid> m_firstAncestorSubgrid;
    SingleThreadWeakPtr<RenderGrid> m_currentAncestorSubgrid;
    const std::optional<GridTrackSizingDirection> m_direction;

};

AncestorSubgridIterator ancestorSubgridsOfGridItem(const RenderBox& gridItem, const GridTrackSizingDirection);

} // namespace WebCore
