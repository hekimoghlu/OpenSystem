/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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
#include "ScrollingConstraints.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AbsolutePositionConstraints);
WTF_MAKE_TZONE_ALLOCATED_IMPL(ViewportConstraints);

AbsolutePositionConstraints::AbsolutePositionConstraints(const FloatSize& alignmentOffset, const FloatPoint& layerPositionAtLastLayout)
    : m_alignmentOffset(alignmentOffset)
    , m_layerPositionAtLastLayout(layerPositionAtLastLayout)
{
}

FloatPoint FixedPositionViewportConstraints::layerPositionForViewportRect(const FloatRect& viewportRect) const
{
    FloatSize offset;

    if (hasAnchorEdge(AnchorEdgeLeft))
        offset.setWidth(viewportRect.x() - m_viewportRectAtLastLayout.x());
    else if (hasAnchorEdge(AnchorEdgeRight))
        offset.setWidth(viewportRect.maxX() - m_viewportRectAtLastLayout.maxX());

    if (hasAnchorEdge(AnchorEdgeTop))
        offset.setHeight(viewportRect.y() - m_viewportRectAtLastLayout.y());
    else if (hasAnchorEdge(AnchorEdgeBottom))
        offset.setHeight(viewportRect.maxY() - m_viewportRectAtLastLayout.maxY());

    return m_layerPositionAtLastLayout + offset;
}

FloatSize StickyPositionViewportConstraints::computeStickyOffset(const FloatRect& constrainingRect) const
{
    FloatRect boxRect = m_stickyBoxRect;
    
    if (hasAnchorEdge(AnchorEdgeRight)) {
        float rightLimit = constrainingRect.maxX() - m_rightOffset;
        float rightDelta = std::min<float>(0, rightLimit - m_stickyBoxRect.maxX());
        float availableSpace = std::min<float>(0, m_containingBlockRect.x() - m_stickyBoxRect.x());
        if (rightDelta < availableSpace)
            rightDelta = availableSpace;

        boxRect.move(rightDelta, 0);
    }

    if (hasAnchorEdge(AnchorEdgeLeft)) {
        float leftLimit = constrainingRect.x() + m_leftOffset;
        float leftDelta = std::max<float>(0, leftLimit - m_stickyBoxRect.x());
        float availableSpace = std::max<float>(0, m_containingBlockRect.maxX() - m_stickyBoxRect.maxX());
        if (leftDelta > availableSpace)
            leftDelta = availableSpace;

        boxRect.move(leftDelta, 0);
    }
    
    if (hasAnchorEdge(AnchorEdgeBottom)) {
        float bottomLimit = constrainingRect.maxY() - m_bottomOffset;
        float bottomDelta = std::min<float>(0, bottomLimit - m_stickyBoxRect.maxY());
        float availableSpace = std::min<float>(0, m_containingBlockRect.y() - m_stickyBoxRect.y());
        if (bottomDelta < availableSpace)
            bottomDelta = availableSpace;

        boxRect.move(0, bottomDelta);
    }

    if (hasAnchorEdge(AnchorEdgeTop)) {
        float topLimit = constrainingRect.y() + m_topOffset;
        float topDelta = std::max<float>(0, topLimit - m_stickyBoxRect.y());
        float availableSpace = std::max<float>(0, m_containingBlockRect.maxY() - m_stickyBoxRect.maxY());
        if (topDelta > availableSpace)
            topDelta = availableSpace;

        boxRect.move(0, topDelta);
    }

    return boxRect.location() - m_stickyBoxRect.location();
}

FloatPoint StickyPositionViewportConstraints::layerPositionForConstrainingRect(const FloatRect& constrainingRect) const
{
    FloatSize offset = computeStickyOffset(constrainingRect);
    return m_layerPositionAtLastLayout + offset - m_stickyOffsetAtLastLayout;
}

TextStream& operator<<(TextStream& ts, ScrollPositioningBehavior behavior)
{
    switch (behavior) {
    case ScrollPositioningBehavior::None: ts << "none"; break;
    case ScrollPositioningBehavior::Stationary: ts << "stationary"; break;
    case ScrollPositioningBehavior::Moves: ts << "moves"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, const AbsolutePositionConstraints& constraints)
{
    ts.dumpProperty("layer-position-at-last-layout", constraints.layerPositionAtLastLayout());

    return ts;
}

TextStream& operator<<(TextStream& ts, const FixedPositionViewportConstraints& constraints)
{
    ts.dumpProperty("viewport-rect-at-last-layout", constraints.viewportRectAtLastLayout());
    ts.dumpProperty("layer-position-at-last-layout", constraints.layerPositionAtLastLayout());

    return ts;
}

TextStream& operator<<(TextStream& ts, const StickyPositionViewportConstraints& constraints)
{
    ts.dumpProperty("sticky-position-at-last-layout", constraints.stickyOffsetAtLastLayout());
    ts.dumpProperty("layer-position-at-last-layout", constraints.layerPositionAtLastLayout());

    ts.dumpProperty("sticky-box-rect", constraints.stickyBoxRect());
    ts.dumpProperty("containing-block-rect", constraints.containingBlockRect());
    ts.dumpProperty("constraining-rect-at-last-layout", constraints.constrainingRectAtLastLayout());

    return ts;
}

} // namespace WebCore
