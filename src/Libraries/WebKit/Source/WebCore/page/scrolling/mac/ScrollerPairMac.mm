/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
#import "config.h"
#import "ScrollerPairMac.h"

#if PLATFORM(MAC)

#import "Logging.h"
#import "ScrollTypesMac.h"
#import "ScrollingTreeFrameScrollingNode.h"
#import <WebCore/FloatPoint.h>
#import <WebCore/IntRect.h>
#import <WebCore/NSScrollerImpDetails.h>
#import <WebCore/PlatformWheelEvent.h>
#import <WebCore/ScrollTypes.h>
#import <WebCore/ScrollableArea.h>
#import <WebCore/ScrollingTreeScrollingNode.h>
#import <pal/spi/mac/NSScrollerImpSPI.h>
#import <wtf/TZoneMallocInlines.h>

@interface WebScrollerImpPairDelegateMac : NSObject <NSScrollerImpPairDelegate> {
    ThreadSafeWeakPtr<WebCore::ScrollerPairMac> _scrollerPair;
}
- (id)initWithScrollerPair:(WebCore::ScrollerPairMac*)scrollerPair;
@end

@implementation WebScrollerImpPairDelegateMac

- (id)initWithScrollerPair:(WebCore::ScrollerPairMac*)scrollerPair
{
    self = [super init];
    if (!self)
        return nil;

    _scrollerPair = scrollerPair;
    return self;
}

- (void)invalidate
{
    _scrollerPair = nullptr;
}

- (NSRect)contentAreaRectForScrollerImpPair:(NSScrollerImpPair *)scrollerImpPair
{
    UNUSED_PARAM(scrollerImpPair);
    RefPtr scrollerPair = _scrollerPair.get();
    if (!scrollerPair)
        return NSZeroRect;

    auto size = scrollerPair->visibleSize();
    return NSMakeRect(0, 0, size.width(), size.height());
}

- (BOOL)inLiveResizeForScrollerImpPair:(NSScrollerImpPair *)scrollerImpPair
{
    RefPtr scrollerPair = _scrollerPair.get();
    if (!scrollerPair)
        return NO;

    return scrollerPair->inLiveResize();
}

- (NSPoint)mouseLocationInContentAreaForScrollerImpPair:(NSScrollerImpPair *)scrollerImpPair
{
    UNUSED_PARAM(scrollerImpPair);
    // This location is only used when calling mouseLocationInScrollerForScrollerImp,
    // where we will use the converted mouse position from the Web Process
    return NSZeroPoint;
}

- (NSPoint)scrollerImpPair:(NSScrollerImpPair *)scrollerImpPair convertContentPoint:(NSPoint)pointInContentArea toScrollerImp:(NSScrollerImp *)scrollerImp
{
    UNUSED_PARAM(scrollerImpPair);
    UNUSED_PARAM(pointInContentArea);

    RefPtr scrollerPair = _scrollerPair.get();
    if (!scrollerPair)
        return NSZeroPoint;

    if (!scrollerPair || !scrollerImp)
        return NSZeroPoint;

    WebCore::ScrollerMac* scroller = nullptr;
    if ([scrollerImp isHorizontal])
        scroller = &scrollerPair->horizontalScroller();
    else
        scroller = &scrollerPair->verticalScroller();

    ASSERT(scrollerImp == scroller->scrollerImp());

    return scroller->lastKnownMousePositionInScrollbar();
}

- (void)scrollerImpPair:(NSScrollerImpPair *)scrollerImpPair setContentAreaNeedsDisplayInRect:(NSRect)rect
{
    UNUSED_PARAM(scrollerImpPair);
    UNUSED_PARAM(rect);
}

- (void)scrollerImpPair:(NSScrollerImpPair *)scrollerImpPair updateScrollerStyleForNewRecommendedScrollerStyle:(NSScrollerStyle)newRecommendedScrollerStyle
{
    UNUSED_PARAM(scrollerImpPair);

    RefPtr scrollerPair = _scrollerPair.get();
    if (scrollerPair)
        scrollerPair->setScrollbarStyle(WebCore::scrollbarStyle(newRecommendedScrollerStyle));
}

@end

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScrollerPairMac);

ScrollerPairMac::ScrollerPairMac(ScrollingTreeScrollingNode& node)
    : m_scrollingNode(node)
    , m_verticalScroller(*this, ScrollbarOrientation::Vertical)
    , m_horizontalScroller(*this, ScrollbarOrientation::Horizontal)
{
}

void ScrollerPairMac::init()
{
    m_scrollerImpPairDelegate = adoptNS([[WebScrollerImpPairDelegateMac alloc] initWithScrollerPair:this]);

    m_scrollerImpPair = adoptNS([[NSScrollerImpPair alloc] init]);
    [m_scrollerImpPair setDelegate:m_scrollerImpPairDelegate.get()];
    auto style = ScrollerStyle::recommendedScrollerStyle();
    m_scrollbarStyle = WebCore::scrollbarStyle(style);
    [m_scrollerImpPair setScrollerStyle:style];

    m_verticalScroller.attach();
    m_horizontalScroller.attach();
}

ScrollerPairMac::~ScrollerPairMac()
{
    [m_scrollerImpPairDelegate invalidate];
    [m_scrollerImpPair setDelegate:nil];
    
    m_verticalScroller.detach();
    m_horizontalScroller.detach();

    ensureOnMainThread([scrollerImpPair = std::exchange(m_scrollerImpPair, nil), verticalScrollerImp = verticalScroller().takeScrollerImp(), horizontalScrollerImp = horizontalScroller().takeScrollerImp()] {
    });
}

void ScrollerPairMac::handleWheelEventPhase(PlatformWheelEventPhase phase)
{
    ensureOnMainThreadWithProtectedThis([phase, this] {
        switch (phase) {
        case PlatformWheelEventPhase::Began:
            [m_scrollerImpPair beginScrollGesture];
            break;
        case PlatformWheelEventPhase::Ended:
        case PlatformWheelEventPhase::Cancelled:
            [m_scrollerImpPair endScrollGesture];
            break;
        case PlatformWheelEventPhase::MayBegin:
            [m_scrollerImpPair beginScrollGesture];
            [m_scrollerImpPair contentAreaScrolled];
            break;
        default:
            break;
        }
    });
}

void ScrollerPairMac::viewWillStartLiveResize()
{
    if (m_inLiveResize)
        return;
    
    m_inLiveResize = true;

    ensureOnMainThreadWithProtectedThis([this] {
        if ([m_scrollerImpPair overlayScrollerStateIsLocked])
            return;

        [m_scrollerImpPair startLiveResize];
    });
}

void ScrollerPairMac::viewWillEndLiveResize()
{
    if (!m_inLiveResize)
        return;
    
    m_inLiveResize = false;

    ensureOnMainThreadWithProtectedThis([this] {
        if ([m_scrollerImpPair overlayScrollerStateIsLocked])
            return;

        [m_scrollerImpPair endLiveResize];
    });
}

void ScrollerPairMac::contentsSizeChanged()
{
    ensureOnMainThreadWithProtectedThis([this] {
        if ([m_scrollerImpPair overlayScrollerStateIsLocked])
            return;

        [m_scrollerImpPair contentAreaDidResize];
    });
}

void ScrollerPairMac::setUsePresentationValues(bool inMomentumPhase)
{
    m_usingPresentationValues = inMomentumPhase;
    [scrollerImpHorizontal() setUsePresentationValue:m_usingPresentationValues];
    [scrollerImpVertical() setUsePresentationValue:m_usingPresentationValues];
}

void ScrollerPairMac::setHorizontalScrollbarPresentationValue(float scrollbValue)
{
    [scrollerImpHorizontal() setPresentationValue:scrollbValue];
}

void ScrollerPairMac::setVerticalScrollbarPresentationValue(float scrollbValue)
{
    [scrollerImpVertical() setPresentationValue:scrollbValue];
}

void ScrollerPairMac::updateValues()
{
    RefPtr node = m_scrollingNode.get();
    if (!node)
        return;

    auto offset = node->currentScrollOffset();

    if (offset != m_lastScrollOffset) {
        if (m_lastScrollOffset) {
            ensureOnMainThreadWithProtectedThis([delta = offset - *m_lastScrollOffset, this] {
                [m_scrollerImpPair contentAreaScrolledInDirection:NSMakePoint(delta.width(), delta.height())];
            });
        }
        m_lastScrollOffset = offset;
    }

    m_horizontalScroller.updateValues();
    m_verticalScroller.updateValues();
}

FloatSize ScrollerPairMac::visibleSize() const
{
    RefPtr node = m_scrollingNode.get();
    if (!node)
        return { };

    return node->scrollableAreaSize();
}

bool ScrollerPairMac::useDarkAppearance() const
{
    return m_useDarkAppearance;
}

ScrollbarWidth ScrollerPairMac::scrollbarWidthStyle() const
{
    return m_scrollbarWidth;
}

ScrollerPairMac::Values ScrollerPairMac::valuesForOrientation(ScrollbarOrientation orientation)
{
    RefPtr node = m_scrollingNode.get();
    if (!node)
        return { };

    float position;
    float totalSize;
    float visibleSize;
    if (orientation == ScrollbarOrientation::Vertical) {
        position = node->currentScrollOffset().y();
        totalSize = node->totalContentsSize().height();
        visibleSize = node->scrollableAreaSize().height();
    } else {
        position = node->currentScrollOffset().x();
        totalSize = node->totalContentsSize().width();
        visibleSize = node->scrollableAreaSize().width();
    }

    float value;
    float overhang;
    ScrollableArea::computeScrollbarValueAndOverhang(position, totalSize, visibleSize, value, overhang);

    float proportion = totalSize ? (visibleSize - overhang) / totalSize : 1;

    return { value, proportion };
}

bool ScrollerPairMac::hasScrollerImp()
{
    return verticalScroller().scrollerImp() || horizontalScroller().scrollerImp();
}

void ScrollerPairMac::releaseReferencesToScrollerImpsOnTheMainThread()
{
    if (hasScrollerImp()) {
        // FIXME: This is a workaround in place for the time being since NSScrollerImps cannot be deallocated
        // on a non-main thread. rdar://problem/24535055
        WTF::callOnMainThread([verticalScrollerImp = verticalScroller().takeScrollerImp(), horizontalScrollerImp = horizontalScroller().takeScrollerImp()] {
        });
    }
}

String ScrollerPairMac::scrollbarStateForOrientation(ScrollbarOrientation orientation) const
{
    return orientation == ScrollbarOrientation::Vertical ? m_verticalScroller.scrollbarState() : m_horizontalScroller.scrollbarState();
}

void ScrollerPairMac::setVerticalScrollerImp(NSScrollerImp *scrollerImp)
{
    ensureOnMainThreadWithProtectedThis([this, scrollerImp = RetainPtr { scrollerImp }] {
        [m_scrollerImpPair setVerticalScrollerImp:scrollerImp.get()];
    });
}

void ScrollerPairMac::setHorizontalScrollerImp(NSScrollerImp *scrollerImp)
{
    ensureOnMainThreadWithProtectedThis([this, scrollerImp = RetainPtr { scrollerImp }] {
        [m_scrollerImpPair setHorizontalScrollerImp:scrollerImp.get()];
    });
}

void ScrollerPairMac::setScrollbarStyle(ScrollbarStyle style)
{
    m_scrollbarStyle = style;

    ensureOnMainThreadWithProtectedThis([this, scrollerStyle = nsScrollerStyle(style)] {
        m_horizontalScroller.updateScrollbarStyle();
        m_verticalScroller.updateScrollbarStyle();
        [m_scrollerImpPair setScrollerStyle:scrollerStyle];
    });
}

void ScrollerPairMac::ensureOnMainThreadWithProtectedThis(Function<void()>&& task)
{
    ensureOnMainThread([protectedThis = Ref { *this }, task = WTFMove(task)]() mutable {
        task();
    });
}

void ScrollerPairMac::mouseEnteredContentArea()
{
    LOG_WITH_STREAM(OverlayScrollbars, stream << "ScrollerPairMac for [" << protectedNode()->scrollingNodeID() << "] mouseEnteredContentArea");

    ensureOnMainThreadWithProtectedThis([this] {
        if ([m_scrollerImpPair overlayScrollerStateIsLocked])
            return;

        [m_scrollerImpPair mouseEnteredContentArea];
    });
}

void ScrollerPairMac::mouseExitedContentArea()
{
    m_mouseInContentArea = false;
    LOG_WITH_STREAM(OverlayScrollbars, stream << "ScrollerPairMac for [" << protectedNode()->scrollingNodeID() << "] mouseExitedContentArea");

    ensureOnMainThreadWithProtectedThis([this] {
        if ([m_scrollerImpPair overlayScrollerStateIsLocked])
            return;

        [m_scrollerImpPair mouseExitedContentArea];
    });
}

void ScrollerPairMac::mouseMovedInContentArea(const MouseLocationState& state)
{
    m_mouseInContentArea = true;
    horizontalScroller().setLastKnownMousePositionInScrollbar(state.locationInHorizontalScrollbar);
    verticalScroller().setLastKnownMousePositionInScrollbar(state.locationInVerticalScrollbar);

    ensureOnMainThreadWithProtectedThis([this] {
        if ([m_scrollerImpPair overlayScrollerStateIsLocked])
            return;
        
        [m_scrollerImpPair mouseMovedInContentArea];
    });
}

void ScrollerPairMac::mouseIsInScrollbar(ScrollbarHoverState hoverState)
{
    if (m_scrollbarHoverState.mouseIsOverVerticalScrollbar != hoverState.mouseIsOverVerticalScrollbar) {
        if (hoverState.mouseIsOverVerticalScrollbar)
            verticalScroller().mouseEnteredScrollbar();
        else
            verticalScroller().mouseExitedScrollbar();
    }

    if (m_scrollbarHoverState.mouseIsOverHorizontalScrollbar != hoverState.mouseIsOverHorizontalScrollbar) {
        if (hoverState.mouseIsOverHorizontalScrollbar)
            horizontalScroller().mouseEnteredScrollbar();
        else
            horizontalScroller().mouseExitedScrollbar();
    }
    m_scrollbarHoverState = hoverState;
}

void ScrollerPairMac::setUseDarkAppearance(bool useDarkAppearance)
{
    if (m_useDarkAppearance == useDarkAppearance)
        return;
    m_useDarkAppearance = useDarkAppearance;

    horizontalScroller().setNeedsDisplay();
    verticalScroller().setNeedsDisplay();
}

void ScrollerPairMac::setScrollbarWidth(ScrollbarWidth scrollbarWidth)
{
    if (m_scrollbarWidth == scrollbarWidth)
        return;
    m_scrollbarWidth = scrollbarWidth;

    horizontalScroller().updateScrollbarStyle();
    verticalScroller().updateScrollbarStyle();
}

} // namespace WebCore

#endif // PLATFORM(MAC)
