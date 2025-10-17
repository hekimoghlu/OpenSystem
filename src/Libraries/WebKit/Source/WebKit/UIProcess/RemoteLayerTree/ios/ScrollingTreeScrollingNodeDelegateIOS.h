/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#if PLATFORM(IOS_FAMILY) && ENABLE(ASYNC_SCROLLING)

#import "WKBrowserEngineDefinitions.h"
#import <UIKit/UIScrollView.h>
#import <WebCore/ScrollingCoordinator.h>
#import <WebCore/ScrollingTreeScrollingNode.h>
#import <WebCore/ScrollingTreeScrollingNodeDelegate.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakPtr.h>

@class CALayer;
@class UIScrollView;
@class WKBaseScrollView;
@class WKBEScrollViewScrollUpdate;
@class WKBEScrollView;
@class WKScrollingNodeScrollViewDelegate;

namespace WebCore {

class FloatPoint;
class FloatRect;
class FloatSize;
class ScrollingTreeScrollingNode;

}

namespace WebKit {

class ScrollingTreeScrollingNodeDelegateIOS final : public WebCore::ScrollingTreeScrollingNodeDelegate, public CanMakeWeakPtr<ScrollingTreeScrollingNodeDelegateIOS>, public CanMakeCheckedPtr<ScrollingTreeScrollingNodeDelegateIOS> {
    WTF_MAKE_TZONE_ALLOCATED(ScrollingTreeScrollingNodeDelegateIOS);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ScrollingTreeScrollingNodeDelegateIOS);
public:
    
    enum class AllowOverscrollToPreventScrollPropagation : bool { No, Yes };
    
    explicit ScrollingTreeScrollingNodeDelegateIOS(WebCore::ScrollingTreeScrollingNode&);
    ~ScrollingTreeScrollingNodeDelegateIOS() final;

    void scrollWillStart() const;
    void scrollDidEnd() const;
    void scrollViewWillStartPanGesture() const;
    void scrollViewDidScroll(const WebCore::FloatPoint& scrollOffset, bool inUserInteraction);

    void currentSnapPointIndicesDidChange(std::optional<unsigned> horizontal, std::optional<unsigned> vertical) const;
    CALayer *scrollLayer() const { return m_scrollLayer.get(); }

    void resetScrollViewDelegate();
    void commitStateBeforeChildren(const WebCore::ScrollingStateScrollingNode&);
    void commitStateAfterChildren(const WebCore::ScrollingStateScrollingNode&);

    void repositionScrollingLayers();

#if HAVE(UISCROLLVIEW_ASYNCHRONOUS_SCROLL_EVENT_HANDLING)
    void handleAsynchronousCancelableScrollEvent(WKBaseScrollView *, WKBEScrollViewScrollUpdate *, void (^completion)(BOOL handled));
#endif

    OptionSet<WebCore::TouchAction> activeTouchActions() const { return m_activeTouchActions; }
    void computeActiveTouchActionsForGestureRecognizer(UIGestureRecognizer*);
    void clearActiveTouchActions() { m_activeTouchActions = { }; }
    void cancelPointersForGestureRecognizer(UIGestureRecognizer*);
    bool shouldAllowPanGestureRecognizerToReceiveTouches() const;

    UIScrollView *findActingScrollParent(UIScrollView *);
    WKBaseScrollView *scrollView() const;

    bool startAnimatedScrollToPosition(WebCore::FloatPoint) final;
    void stopAnimatedScroll() final;

    void serviceScrollAnimation(MonotonicTime) final { }
    
    static void updateScrollViewForOverscrollBehavior(UIScrollView *, const WebCore::OverscrollBehavior, const WebCore::OverscrollBehavior, AllowOverscrollToPreventScrollPropagation);

private:
    RetainPtr<CALayer> m_scrollLayer;
#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    RetainPtr<CALayer> m_interactionRegionsLayer;
#endif
    RetainPtr<CALayer> m_scrolledContentsLayer;
    RetainPtr<WKScrollingNodeScrollViewDelegate> m_scrollViewDelegate;
    OptionSet<WebCore::TouchAction> m_activeTouchActions { };
    bool m_updatingFromStateNode { false };
};

} // namespace WebKit

@interface WKScrollingNodeScrollViewDelegate : NSObject <WKBEScrollViewDelegate> {
    WeakPtr<WebKit::ScrollingTreeScrollingNodeDelegateIOS> _scrollingTreeNodeDelegate;
}

@property (nonatomic, getter=_isInUserInteraction) BOOL inUserInteraction;

- (instancetype)initWithScrollingTreeNodeDelegate:(WebKit::ScrollingTreeScrollingNodeDelegateIOS&)delegate;

@end

#endif // PLATFORM(IOS_FAMILY) && ENABLE(ASYNC_SCROLLING)
