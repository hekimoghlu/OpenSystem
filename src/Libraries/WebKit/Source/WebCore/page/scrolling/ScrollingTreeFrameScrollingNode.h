/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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

#if ENABLE(ASYNC_SCROLLING)

#include "ScrollingTreeScrollingNode.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class PlatformWheelEvent;
class ScrollingTree;

class WEBCORE_EXPORT ScrollingTreeFrameScrollingNode : public ScrollingTreeScrollingNode {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollingTreeFrameScrollingNode, WEBCORE_EXPORT);
public:
    virtual ~ScrollingTreeFrameScrollingNode();

    bool commitStateBeforeChildren(const ScrollingStateNode&) override;
    
    bool visualViewportIsSmallerThanLayoutViewport() const { return m_visualViewportIsSmallerThanLayoutViewport; }

    FloatSize viewToContentsOffset(const FloatPoint& scrollPosition) const;

    FloatRect layoutViewport() const { return m_layoutViewport; };
    void setLayoutViewport(const FloatRect& r) { m_layoutViewport = r; };

    FloatRect layoutViewportRespectingRubberBanding() const;

    float frameScaleFactor() const { return m_frameScaleFactor; }
    int headerHeight() const { return m_headerHeight; }
    int footerHeight() const { return m_footerHeight; }
    float topContentInset() const { return m_topContentInset; }
    virtual void viewWillStartLiveResize() { }
    virtual void viewWillEndLiveResize() { }
    virtual void viewSizeDidChange() { }
protected:
    ScrollingTreeFrameScrollingNode(ScrollingTree&, ScrollingNodeType, ScrollingNodeID);

    FloatPoint minLayoutViewportOrigin() const { return m_minLayoutViewportOrigin; }
    FloatPoint maxLayoutViewportOrigin() const { return m_maxLayoutViewportOrigin; }

    ScrollBehaviorForFixedElements scrollBehaviorForFixedElements() const { return m_behaviorForFixed; }

private:
    void updateViewportForCurrentScrollPosition(std::optional<FloatRect>) override;
    bool scrollPositionAndLayoutViewportMatch(const FloatPoint& position, std::optional<FloatRect> overrideLayoutViewport) override;
    FloatRect layoutViewportForScrollPosition(const FloatPoint&, float scale, ScrollBehaviorForFixedElements = ScrollBehaviorForFixedElements::StickToDocumentBounds) const;

    void dumpProperties(WTF::TextStream&, OptionSet<ScrollingStateTreeAsTextBehavior>) const override;

    FloatRect m_layoutViewport;
    FloatPoint m_minLayoutViewportOrigin;
    FloatPoint m_maxLayoutViewportOrigin;
    std::optional<FloatSize> m_overrideVisualViewportSize;
    
    float m_frameScaleFactor { 1 };
    float m_topContentInset { 0 };

    int m_headerHeight { 0 };
    int m_footerHeight { 0 };
    
    ScrollBehaviorForFixedElements m_behaviorForFixed { ScrollBehaviorForFixedElements::StickToDocumentBounds };
    
    bool m_visualViewportIsSmallerThanLayoutViewport { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_SCROLLING_NODE(ScrollingTreeFrameScrollingNode, isFrameScrollingNode())

#endif // ENABLE(ASYNC_SCROLLING)
