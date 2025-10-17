/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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

#if PLATFORM(MAC)

#include "FloatRect.h"
#include "FloatSize.h"
#include "PlatformWheelEvent.h"
#include "ScrollerMac.h"
#include "ScrollingStateScrollingNode.h"
#include <wtf/RecursiveLockAdapter.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

OBJC_CLASS NSScrollerImp;
OBJC_CLASS NSScrollerImpPair;
OBJC_CLASS WebScrollerImpPairDelegateMac;

namespace WebCore {
class PlatformWheelEvent;
class ScrollingTreeScrollingNode;
}

namespace WebCore {

// Controls a pair of NSScrollerImps via a pair of ScrollerMac. The NSScrollerImps need to remain internal to this class.
class ScrollerPairMac : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<ScrollerPairMac> {
    WTF_MAKE_TZONE_ALLOCATED(ScrollerPairMac);
    friend class ScrollerMac;
public:
    void init();

    static Ref<ScrollerPairMac> create(ScrollingTreeScrollingNode& node)
    {
        return adoptRef(*new ScrollerPairMac(node));
    }

    ~ScrollerPairMac();

    void handleWheelEventPhase(PlatformWheelEventPhase);

    void setUsePresentationValues(bool);
    bool isUsingPresentationValues() const { return m_usingPresentationValues; }

    void setVerticalScrollbarPresentationValue(float);
    void setHorizontalScrollbarPresentationValue(float);

    void updateValues();

    FloatSize visibleSize() const;
    bool useDarkAppearance() const;
    ScrollbarWidth scrollbarWidthStyle() const;

    struct Values {
        float value;
        float proportion;
    };
    Values valuesForOrientation(ScrollbarOrientation);

    void releaseReferencesToScrollerImpsOnTheMainThread();

    bool hasScrollerImp();

    void viewWillStartLiveResize();
    void viewWillEndLiveResize();
    void contentsSizeChanged();
    bool inLiveResize() const { return m_inLiveResize; }

    // Only for use by WebScrollerImpPairDelegateMac. Do not use elsewhere!
    ScrollerMac& verticalScroller() { return m_verticalScroller; }
    ScrollerMac& horizontalScroller() { return m_horizontalScroller; }

    String scrollbarStateForOrientation(ScrollbarOrientation) const;

    ScrollbarStyle scrollbarStyle() const { return m_scrollbarStyle; }
    void setScrollbarStyle(ScrollbarStyle);

    void setVerticalScrollerImp(NSScrollerImp *);
    void setHorizontalScrollerImp(NSScrollerImp *);
    
    void mouseEnteredContentArea();
    void mouseExitedContentArea();
    void mouseMovedInContentArea(const MouseLocationState&);
    void mouseIsInScrollbar(ScrollbarHoverState);

    NSScrollerImpPair *scrollerImpPair() const { return m_scrollerImpPair.get(); }
    void ensureOnMainThreadWithProtectedThis(Function<void()>&&);
    RefPtr<ScrollingTreeScrollingNode> protectedNode() const { return m_scrollingNode.get(); }

    bool mouseInContentArea() const { return m_mouseInContentArea; }

    void setUseDarkAppearance(bool);

    void setScrollbarWidth(ScrollbarWidth);

private:
    ScrollerPairMac(ScrollingTreeScrollingNode&);

    NSScrollerImp *scrollerImpHorizontal() { return horizontalScroller().scrollerImp(); }
    NSScrollerImp *scrollerImpVertical() { return verticalScroller().scrollerImp(); }

    ThreadSafeWeakPtr<ScrollingTreeScrollingNode> m_scrollingNode;

    ScrollbarHoverState m_scrollbarHoverState;

    ScrollerMac m_verticalScroller;
    ScrollerMac m_horizontalScroller;

    std::optional<FloatPoint> m_lastScrollOffset;

    RetainPtr<NSScrollerImpPair> m_scrollerImpPair;
    RetainPtr<WebScrollerImpPairDelegateMac> m_scrollerImpPairDelegate;

    ScrollbarWidth m_scrollbarWidth { ScrollbarWidth::Auto };
    std::atomic<bool> m_usingPresentationValues { false };
    std::atomic<ScrollbarStyle> m_scrollbarStyle { ScrollbarStyle::AlwaysVisible };
    bool m_useDarkAppearance { false };
    bool m_inLiveResize { false };
    bool m_mouseInContentArea { false };
};

}

#endif
