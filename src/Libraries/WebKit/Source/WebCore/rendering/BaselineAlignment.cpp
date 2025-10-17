/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#include "BaselineAlignment.h"

#include "BaselineAlignmentInlines.h"
#include "RenderBox.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BaselineGroup);
WTF_MAKE_TZONE_ALLOCATED_IMPL(BaselineAlignmentState);

BaselineGroup::BaselineGroup(FlowDirection blockFlow, ItemPosition childPreference)
    : m_maxAscent(0), m_items()
{
    m_blockFlow = blockFlow;
    m_preference = childPreference;
}

void BaselineGroup::update(const RenderBox& child, LayoutUnit ascent)
{
    if (m_items.add(child).isNewEntry)
        m_maxAscent = std::max(m_maxAscent, ascent);
}

bool BaselineGroup::isOppositeBlockFlow(FlowDirection blockFlow) const
{
    switch (blockFlow) {
    case FlowDirection::TopToBottom:
        return false;
    case FlowDirection::LeftToRight:
        return m_blockFlow == FlowDirection::RightToLeft;
    case FlowDirection::RightToLeft:
        return m_blockFlow == FlowDirection::LeftToRight;
    default:
        ASSERT_NOT_REACHED();
        return false;
    }
}

bool BaselineGroup::isOrthogonalBlockFlow(FlowDirection blockFlow) const
{
    switch (blockFlow) {
    case FlowDirection::TopToBottom:
        return m_blockFlow != FlowDirection::TopToBottom;
    case FlowDirection::LeftToRight:
    case FlowDirection::RightToLeft:
        return m_blockFlow == FlowDirection::TopToBottom;
    default:
        ASSERT_NOT_REACHED();
        return false;
    }
}

bool BaselineGroup::isCompatible(FlowDirection childBlockFlow, ItemPosition childPreference) const
{
    ASSERT(isBaselinePosition(childPreference));
    ASSERT(computeSize() > 0);
    return ((m_blockFlow == childBlockFlow || isOrthogonalBlockFlow(childBlockFlow)) && m_preference == childPreference) || (isOppositeBlockFlow(childBlockFlow) && m_preference != childPreference);
}

BaselineAlignmentState::BaselineAlignmentState(const RenderBox& child, ItemPosition preference, LayoutUnit ascent)
{
    ASSERT(isBaselinePosition(preference));
    updateSharedGroup(child, preference, ascent);
}

const BaselineGroup& BaselineAlignmentState::sharedGroup(const RenderBox& child, ItemPosition preference) const
{
    ASSERT(isBaselinePosition(preference));
    return const_cast<BaselineAlignmentState*>(this)->findCompatibleSharedGroup(child, preference);
}

Vector<BaselineGroup>& BaselineAlignmentState::sharedGroups()
{
    return m_sharedGroups;
}

void BaselineAlignmentState::updateSharedGroup(const RenderBox& child, ItemPosition preference, LayoutUnit ascent)
{
    ASSERT(isBaselinePosition(preference));
    BaselineGroup& group = findCompatibleSharedGroup(child, preference);
    group.update(child, ascent);
}

// FIXME: Properly implement baseline-group compatibility.
// See https://github.com/w3c/csswg-drafts/issues/721
BaselineGroup& BaselineAlignmentState::findCompatibleSharedGroup(const RenderBox& child, ItemPosition preference)
{
    auto blockFlowDirection = child.writingMode().blockDirection();
    for (auto& group : m_sharedGroups) {
        if (group.isCompatible(blockFlowDirection, preference))
            return group;
    }
    m_sharedGroups.insert(0, BaselineGroup(blockFlowDirection, preference));
    return m_sharedGroups[0];
}

} // namespace WebCore
