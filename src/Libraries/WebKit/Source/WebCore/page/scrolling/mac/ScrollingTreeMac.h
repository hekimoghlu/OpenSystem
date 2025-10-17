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
#pragma once

#if ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)

#include "ThreadedScrollingTree.h"

namespace WebCore {

class WheelEventTestMonitor;

class ScrollingTreeMac final : public ThreadedScrollingTree {
public:
    static Ref<ScrollingTreeMac> create(AsyncScrollingCoordinator&);

    void didCompletePlatformRenderingUpdate();

private:
    explicit ScrollingTreeMac(AsyncScrollingCoordinator&);
    
    bool isScrollingTreeMac() const final { return true; }

    Ref<ScrollingTreeNode> createScrollingTreeNode(ScrollingNodeType, ScrollingNodeID) final;

    RefPtr<ScrollingTreeNode> scrollingNodeForPoint(FloatPoint) final;
#if ENABLE(WHEEL_EVENT_REGIONS)
    OptionSet<EventListenerRegionType> eventListenerRegionTypesForPoint(FloatPoint) const final;
#endif

    void registerForPlatformRenderingUpdateCallback();
    void applyLayerPositionsInternal() final WTF_REQUIRES_LOCK(m_treeLock);

    void didCompleteRenderingUpdate() final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_TREE(WebCore::ScrollingTreeMac, isScrollingTreeMac())

#endif // ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)
