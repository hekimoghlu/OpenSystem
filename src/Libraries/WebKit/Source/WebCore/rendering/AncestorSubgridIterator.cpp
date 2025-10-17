/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
#include "AncestorSubgridIterator.h"

#include "RenderIterator.h"

namespace WebCore {

AncestorSubgridIterator::AncestorSubgridIterator() = default;

AncestorSubgridIterator::AncestorSubgridIterator(SingleThreadWeakPtr<RenderGrid> firstAncestorSubgrid, GridTrackSizingDirection direction)
    : m_firstAncestorSubgrid(firstAncestorSubgrid)
    , m_direction(direction)
{
    ASSERT_IMPLIES(firstAncestorSubgrid,  firstAncestorSubgrid->isSubgrid(direction));
}


AncestorSubgridIterator::AncestorSubgridIterator(SingleThreadWeakPtr<RenderGrid> firstAncestorSubgrid, SingleThreadWeakPtr<RenderGrid> currentAncestorSubgrid, GridTrackSizingDirection direction)
    : m_firstAncestorSubgrid(firstAncestorSubgrid)
    , m_currentAncestorSubgrid(currentAncestorSubgrid)
    , m_direction(direction)
{
    ASSERT_IMPLIES(firstAncestorSubgrid, firstAncestorSubgrid->isSubgrid(direction));
}

AncestorSubgridIterator::AncestorSubgridIterator(SingleThreadWeakPtr<RenderGrid> firstAncestorSubgrid, SingleThreadWeakPtr<RenderGrid> currentAncestorSubgrid, std::optional<GridTrackSizingDirection> direction)
    : m_firstAncestorSubgrid(firstAncestorSubgrid)
    , m_currentAncestorSubgrid(currentAncestorSubgrid)
    , m_direction(direction)
{
}

AncestorSubgridIterator AncestorSubgridIterator::begin()
{
    return AncestorSubgridIterator(m_firstAncestorSubgrid, m_firstAncestorSubgrid, m_direction);
}

AncestorSubgridIterator AncestorSubgridIterator::end()
{
    return AncestorSubgridIterator(m_firstAncestorSubgrid, nullptr, m_direction);
}

AncestorSubgridIterator& AncestorSubgridIterator::operator++()
{
    ASSERT(m_firstAncestorSubgrid && m_currentAncestorSubgrid && m_direction);

    if (m_firstAncestorSubgrid && m_currentAncestorSubgrid && m_direction) {
        auto nextAncestor = RenderTraversal::findAncestorOfType<RenderGrid>(*m_currentAncestorSubgrid);
        m_currentAncestorSubgrid = (nextAncestor && nextAncestor->isSubgrid(GridLayoutFunctions::flowAwareDirectionForGridItem(*nextAncestor, *m_firstAncestorSubgrid, m_direction.value()))) ? nextAncestor : nullptr;
    }
    return *this;
}

RenderGrid& AncestorSubgridIterator::operator*()
{
    ASSERT(m_currentAncestorSubgrid);
    return *m_currentAncestorSubgrid;
}

bool AncestorSubgridIterator::operator==(const AncestorSubgridIterator& other) const
{
    return m_currentAncestorSubgrid == other.m_currentAncestorSubgrid && m_firstAncestorSubgrid == other.m_firstAncestorSubgrid && m_direction == other.m_direction;
}

AncestorSubgridIterator ancestorSubgridsOfGridItem(const RenderBox& gridItem,  const GridTrackSizingDirection direction)
{
    ASSERT(gridItem.parent()->isRenderGrid());
    if (const auto* gridItemParent = dynamicDowncast<RenderGrid>(gridItem.parent()); gridItemParent && gridItemParent->isSubgrid(direction))
        return AncestorSubgridIterator(gridItemParent, direction);
    return AncestorSubgridIterator();
}

} // namespace WebCore
