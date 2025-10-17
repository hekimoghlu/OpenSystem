/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
#include "ScrollingTreeCoordinated.h"

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "AsyncScrollingCoordinator.h"
#include "CoordinatedPlatformLayer.h"
#include "ScrollingThread.h"
#include "ScrollingTreeFixedNodeCoordinated.h"
#include "ScrollingTreeFrameHostingNode.h"
#include "ScrollingTreeFrameScrollingNodeCoordinated.h"
#include "ScrollingTreeOverflowScrollProxyNodeCoordinated.h"
#include "ScrollingTreeOverflowScrollingNodeCoordinated.h"
#include "ScrollingTreePositionedNodeCoordinated.h"
#include "ScrollingTreeStickyNodeCoordinated.h"

namespace WebCore {

Ref<ScrollingTreeCoordinated> ScrollingTreeCoordinated::create(AsyncScrollingCoordinator& scrollingCoordinator)
{
    return adoptRef(*new ScrollingTreeCoordinated(scrollingCoordinator));
}

ScrollingTreeCoordinated::ScrollingTreeCoordinated(AsyncScrollingCoordinator& scrollingCoordinator)
    : ThreadedScrollingTree(scrollingCoordinator)
{
}

Ref<ScrollingTreeNode> ScrollingTreeCoordinated::createScrollingTreeNode(ScrollingNodeType nodeType, ScrollingNodeID nodeID)
{
    switch (nodeType) {
    case ScrollingNodeType::MainFrame:
    case ScrollingNodeType::Subframe:
        return ScrollingTreeFrameScrollingNodeCoordinated::create(*this, nodeType, nodeID);
    case ScrollingNodeType::FrameHosting:
        return ScrollingTreeFrameHostingNode::create(*this, nodeID);
    case ScrollingNodeType::Overflow:
        return ScrollingTreeOverflowScrollingNodeCoordinated::create(*this, nodeID);
    case ScrollingNodeType::OverflowProxy:
        return ScrollingTreeOverflowScrollProxyNodeCoordinated::create(*this, nodeID);
    case ScrollingNodeType::Fixed:
        return ScrollingTreeFixedNodeCoordinated::create(*this, nodeID);
    case ScrollingNodeType::Sticky:
        return ScrollingTreeStickyNodeCoordinated::create(*this, nodeID);
    case ScrollingNodeType::Positioned:
        return ScrollingTreePositionedNodeCoordinated::create(*this, nodeID);
    case ScrollingNodeType::PluginScrolling:
    case ScrollingNodeType::PluginHosting:
        RELEASE_ASSERT_NOT_REACHED();
    }

    RELEASE_ASSERT_NOT_REACHED();
}

void ScrollingTreeCoordinated::applyLayerPositionsInternal()
{
    auto* rootScrollingNode = rootNode();
    if (!rootScrollingNode)
        return;

    ThreadedScrollingTree::applyLayerPositionsInternal();

    if (ScrollingThread::isCurrentThread()) {
        auto rootContentsLayer = static_cast<ScrollingTreeFrameScrollingNodeCoordinated*>(rootScrollingNode)->rootContentsLayer();
        rootContentsLayer->requestComposition();
    }
}

void ScrollingTreeCoordinated::didCompleteRenderingUpdate()
{
    // If there's a composition requested or ongoing, wait for didCompletePlatformRenderingUpdate() that will be
    // called once the composiiton finishes.
    if (auto* rootScrollingNode = rootNode()) {
        auto rootContentsLayer = static_cast<ScrollingTreeFrameScrollingNodeCoordinated*>(rootScrollingNode)->rootContentsLayer();
        if (rootContentsLayer->isCompositionRequiredOrOngoing())
            return;
    }

    renderingUpdateComplete();
}

void ScrollingTreeCoordinated::didCompletePlatformRenderingUpdate()
{
    renderingUpdateComplete();
}

static bool collectDescendantLayersAtPoint(Vector<Ref<CoordinatedPlatformLayer>>& layersAtPoint, const Ref<CoordinatedPlatformLayer>& parent, const FloatPoint& point)
{
    bool existsOnDescendent = false;
    bool existsOnLayer = !!parent->scrollingNodeID() && parent->bounds().contains(point) && parent->eventRegion().contains(roundedIntPoint(point));
    for (auto& child : parent->children()) {
        Locker childLocker { child->lock() };
        FloatPoint transformedPoint(point);
        if (child->transform().isInvertible()) {
            float originX = child->anchorPoint().x() * child->size().width();
            float originY = child->anchorPoint().y() * child->size().height();
            auto transform = *(TransformationMatrix()
                .translate3d(originX + child->position().x() - parent->boundsOrigin().x(), originY + child->position().y() - parent->boundsOrigin().y(), child->anchorPoint().z())
                .multiply(child->transform())
                .translate3d(-originX, -originY, -child->anchorPoint().z()).inverse());
            auto pointInChildSpace = transform.projectPoint(point);
            transformedPoint.set(pointInChildSpace.x(), pointInChildSpace.y());
        }
        existsOnDescendent |= collectDescendantLayersAtPoint(layersAtPoint, child, transformedPoint);
    }

    if (existsOnLayer && !existsOnDescendent)
        layersAtPoint.append(parent);

    return existsOnLayer || existsOnDescendent;
}

RefPtr<ScrollingTreeNode> ScrollingTreeCoordinated::scrollingNodeForPoint(FloatPoint point)
{
    auto* rootScrollingNode = rootNode();
    if (!rootScrollingNode)
        return nullptr;

    Locker layerLocker { m_layerHitTestMutex };

    auto rootContentsLayer = static_cast<ScrollingTreeFrameScrollingNodeCoordinated*>(rootScrollingNode)->rootContentsLayer();
    Vector<Ref<CoordinatedPlatformLayer>> layersAtPoint;
    {
        Locker rootContentsLayerLocker { rootContentsLayer->lock() };
        collectDescendantLayersAtPoint(layersAtPoint, Ref { *rootContentsLayer }, point);
    }

    for (auto& layer : makeReversedRange(layersAtPoint)) {
        Locker locker { layer->lock() };
        auto* scrollingNode = nodeForID(layer->scrollingNodeID());
        if (is<ScrollingTreeScrollingNode>(scrollingNode))
            return scrollingNode;
    }

    return rootScrollingNode;
}

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
