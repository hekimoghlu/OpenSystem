/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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

#if ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
#include "ScrollingTreeFrameScrollingNode.h"
#include <wtf/RefPtr.h>

namespace WebCore {
class CoordinatedPlatformLayer;
class ScrollingTreeScrollingNodeDelegateCoordinated;

class ScrollingTreeFrameScrollingNodeCoordinated final : public ScrollingTreeFrameScrollingNode {
public:
    static Ref<ScrollingTreeFrameScrollingNode> create(ScrollingTree&, ScrollingNodeType, ScrollingNodeID);
    virtual ~ScrollingTreeFrameScrollingNodeCoordinated();

    RefPtr<CoordinatedPlatformLayer> rootContentsLayer() const { return m_rootContentsLayer; }

private:
    ScrollingTreeFrameScrollingNodeCoordinated(ScrollingTree&, ScrollingNodeType, ScrollingNodeID);

    ScrollingTreeScrollingNodeDelegateCoordinated& delegate() const;

    bool commitStateBeforeChildren(const ScrollingStateNode&) override;

    WheelEventHandlingResult handleWheelEvent(const PlatformWheelEvent&, EventTargeting) override;
    void currentScrollPositionChanged(ScrollType, ScrollingLayerPositionAction) override;
    void repositionScrollingLayers() override;
    void repositionRelatedLayers() override;

    RefPtr<CoordinatedPlatformLayer> m_rootContentsLayer;
    RefPtr<CoordinatedPlatformLayer> m_counterScrollingLayer;
    RefPtr<CoordinatedPlatformLayer> m_insetClipLayer;
    RefPtr<CoordinatedPlatformLayer> m_contentShadowLayer;
    RefPtr<CoordinatedPlatformLayer> m_headerLayer;
    RefPtr<CoordinatedPlatformLayer> m_footerLayer;
};

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && USE(COORDINATED_GRAPHICS)
