/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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

class ScrollingTreeScrollingNodeDelegate {
    WTF_MAKE_TZONE_ALLOCATED(ScrollingTreeScrollingNodeDelegate);
public:
    WEBCORE_EXPORT explicit ScrollingTreeScrollingNodeDelegate(ScrollingTreeScrollingNode&);
    WEBCORE_EXPORT virtual ~ScrollingTreeScrollingNodeDelegate();

    ScrollingTreeScrollingNode& scrollingNode() { return m_scrollingNode; }
    const ScrollingTreeScrollingNode& scrollingNode() const { return m_scrollingNode; }
    Ref<ScrollingTreeScrollingNode> protectedScrollingNode() const { return m_scrollingNode; }
    
    virtual bool startAnimatedScrollToPosition(FloatPoint) = 0;
    virtual void stopAnimatedScroll() = 0;

    virtual void serviceScrollAnimation(MonotonicTime) = 0;

    virtual void updateFromStateNode(const ScrollingStateScrollingNode&) { }
    
    virtual void handleWheelEventPhase(const PlatformWheelEventPhase) { }
    
    virtual void viewWillStartLiveResize() { }
    virtual void viewWillEndLiveResize() { }
    virtual void viewSizeDidChange() { }
    
    virtual void updateScrollbarLayers() { }
    virtual void initScrollbars() { }

    virtual void handleKeyboardScrollRequest(const RequestedKeyboardScrollData&) { }

    virtual FloatPoint adjustedScrollPosition(const FloatPoint& scrollPosition) const { return scrollPosition; }
    virtual String scrollbarStateForOrientation(ScrollbarOrientation) const { return ""_s; }

protected:
    WEBCORE_EXPORT RefPtr<ScrollingTree> scrollingTree() const;

    WEBCORE_EXPORT FloatPoint lastCommittedScrollPosition() const;
    WEBCORE_EXPORT FloatSize totalContentsSize();
    WEBCORE_EXPORT FloatSize reachableContentsSize();
    WEBCORE_EXPORT IntPoint scrollOrigin() const;

    FloatPoint currentScrollPosition() const { return m_scrollingNode.currentScrollPosition(); }
    FloatPoint minimumScrollPosition() const { return m_scrollingNode.minimumScrollPosition(); }
    FloatPoint maximumScrollPosition() const { return m_scrollingNode.maximumScrollPosition(); }

    FloatSize scrollableAreaSize() const { return m_scrollingNode.scrollableAreaSize(); }
    FloatSize totalContentsSize() const { return m_scrollingNode.totalContentsSize(); }

    bool allowsHorizontalScrolling() const { return m_scrollingNode.allowsHorizontalScrolling(); }
    bool allowsVerticalScrolling() const { return m_scrollingNode.allowsVerticalScrolling(); }

    ScrollElasticity horizontalScrollElasticity() const { return m_scrollingNode.horizontalScrollElasticity(); }
    ScrollElasticity verticalScrollElasticity() const { return m_scrollingNode.verticalScrollElasticity(); }

private:
    ScrollingTreeScrollingNode& m_scrollingNode; // FIXME : Should use a smart pointer.
};

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING)
