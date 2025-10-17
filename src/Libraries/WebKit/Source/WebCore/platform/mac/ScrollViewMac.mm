/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
#import "ScrollView.h"

#if PLATFORM(MAC)

#import "FloatRect.h"
#import "FloatSize.h"
#import "IntRect.h"
#import "Logging.h"
#import "NotImplemented.h"
#import "WebCoreFrameView.h"
#import <wtf/BlockObjCExceptions.h>

@interface NSScrollView ()
- (NSEdgeInsets)contentInsets;
@end

@interface NSWindow (WebWindowDetails)
- (BOOL)_needsToResetDragMargins;
- (void)_setNeedsToResetDragMargins:(BOOL)needs;
@end

namespace WebCore {

inline NSScrollView<WebCoreFrameScrollView> *ScrollView::scrollView() const
{
    ASSERT(!platformWidget() || [platformWidget() isKindOfClass:[NSScrollView class]]);
    ASSERT(!platformWidget() || [platformWidget() conformsToProtocol:@protocol(WebCoreFrameScrollView)]);
    return static_cast<NSScrollView<WebCoreFrameScrollView> *>(platformWidget());
}

NSView *ScrollView::documentView() const
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    return [scrollView() documentView];
    END_BLOCK_OBJC_EXCEPTIONS
    return nil;
}

void ScrollView::platformAddChild(Widget* child)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    NSView *parentView = documentView();
    NSView *childView = child->getOuterView();
    ASSERT(![parentView isDescendantOf:childView]);
    
    // Suppress the resetting of drag margins since we know we can't affect them.
    NSWindow *window = [parentView window];
    BOOL resetDragMargins = [window _needsToResetDragMargins];
    [window _setNeedsToResetDragMargins:NO];
    if ([childView superview] != parentView)
        [parentView addSubview:childView];
    [window _setNeedsToResetDragMargins:resetDragMargins];
    END_BLOCK_OBJC_EXCEPTIONS
}

void ScrollView::platformRemoveChild(Widget* child)
{
    child->removeFromSuperview();
}

void ScrollView::platformSetScrollbarModes()
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [scrollView() setScrollingModes:m_horizontalScrollbarMode vertical:m_verticalScrollbarMode andLock:NO];
    END_BLOCK_OBJC_EXCEPTIONS
}

void ScrollView::platformScrollbarModes(ScrollbarMode& horizontal, ScrollbarMode& vertical) const
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [scrollView() scrollingModes:&horizontal vertical:&vertical];
    END_BLOCK_OBJC_EXCEPTIONS
}

void ScrollView::platformSetCanBlitOnScroll(bool canBlitOnScroll)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [[scrollView() contentView] setCopiesOnScroll:canBlitOnScroll];
ALLOW_DEPRECATED_DECLARATIONS_END
    END_BLOCK_OBJC_EXCEPTIONS
}

bool ScrollView::platformCanBlitOnScroll() const
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return [[scrollView() contentView] copiesOnScroll];
ALLOW_DEPRECATED_DECLARATIONS_END
}

float ScrollView::platformTopContentInset() const
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    return scrollView().contentInsets.top;
    END_BLOCK_OBJC_EXCEPTIONS

    return 0;
}

void ScrollView::platformSetTopContentInset(float topContentInset)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    if (topContentInset)
        scrollView().automaticallyAdjustsContentInsets = NO;
    else
        scrollView().automaticallyAdjustsContentInsets = YES;

    NSEdgeInsets contentInsets = scrollView().contentInsets;
    contentInsets.top = topContentInset;
    scrollView().contentInsets = contentInsets;
    END_BLOCK_OBJC_EXCEPTIONS
}

IntRect ScrollView::platformVisibleContentRect(bool includeScrollbars) const
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    IntRect visibleContentRect = platformVisibleContentRectIncludingObscuredArea(includeScrollbars);

    visibleContentRect.move(scrollView().contentInsets.left, scrollView().contentInsets.top);
    visibleContentRect.contract(scrollView().contentInsets.left + scrollView().contentInsets.right, scrollView().contentInsets.top + scrollView().contentInsets.bottom);

    return visibleContentRect;
    END_BLOCK_OBJC_EXCEPTIONS

    return IntRect();
}

IntSize ScrollView::platformVisibleContentSize(bool includeScrollbars) const
{
    return platformVisibleContentRect(includeScrollbars).size();
}

IntRect ScrollView::platformVisibleContentRectIncludingObscuredArea(bool includeScrollbars) const
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    IntRect visibleContentRectIncludingObscuredArea = enclosingIntRect([scrollView() documentVisibleRect]);

    if (includeScrollbars) {
        IntSize frameSize = IntSize([scrollView() frame].size);
        visibleContentRectIncludingObscuredArea.setSize(frameSize);
    }

    return visibleContentRectIncludingObscuredArea;
    END_BLOCK_OBJC_EXCEPTIONS

    return IntRect();
}

IntSize ScrollView::platformVisibleContentSizeIncludingObscuredArea(bool includeScrollbars) const
{
    return platformVisibleContentRectIncludingObscuredArea(includeScrollbars).size();
}

IntRect ScrollView::platformUnobscuredContentRect(VisibleContentRectIncludesScrollbars scrollbarInclusion) const
{
    return unobscuredContentRectInternal(scrollbarInclusion);
}

void ScrollView::platformSetContentsSize()
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    int w = m_contentsSize.width();
    int h = m_contentsSize.height();
    LOG(Frames, "%p %@ at w %d h %d\n", documentView(), [(id)[documentView() class] className], w, h);            
    [documentView() setFrameSize:NSMakeSize(std::max(0, w), std::max(0, h))];
    END_BLOCK_OBJC_EXCEPTIONS
}

void ScrollView::platformSetScrollbarsSuppressed(bool repaintOnUnsuppress)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [scrollView() setScrollBarsSuppressed:m_scrollbarsSuppressed
                      repaintOnUnsuppress:repaintOnUnsuppress];
    END_BLOCK_OBJC_EXCEPTIONS
}

void ScrollView::platformSetScrollPosition(const IntPoint& scrollPoint)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    NSPoint floatPoint = scrollPoint;
    NSPoint tempPoint = { std::max(-[scrollView() scrollOrigin].x, floatPoint.x), std::max(-[scrollView() scrollOrigin].y, floatPoint.y) };  // Don't use NSMakePoint to work around 4213314.

    // AppKit has the inset factored into all of its scroll positions. In WebCore, we use positions that ignore
    // the insets so that they are equivalent whether or not there is an inset.
    tempPoint.x = tempPoint.x - scrollView().contentInsets.left;
    tempPoint.y = tempPoint.y - scrollView().contentInsets.top;

    [documentView() scrollPoint:tempPoint];
    END_BLOCK_OBJC_EXCEPTIONS
}

bool ScrollView::platformScroll(ScrollDirection, ScrollGranularity)
{
    // FIXME: It would be nice to implement this so that all of the code in WebFrameView could go away.
    notImplemented();
    return false;
}

void ScrollView::platformRepaintContentRectangle(const IntRect& rect)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    NSView *view = documentView();
    [view setNeedsDisplayInRect:rect];

    END_BLOCK_OBJC_EXCEPTIONS
}

// "Containing Window" means the NSWindow's coord system, which is origin lower left

IntRect ScrollView::platformContentsToScreen(const IntRect& rect) const
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    if (NSView* documentView = this->documentView()) {
        NSRect tempRect = rect;
        tempRect = [documentView convertRect:tempRect toView:nil];
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        tempRect.origin = [[documentView window] convertBaseToScreen:tempRect.origin];
ALLOW_DEPRECATED_DECLARATIONS_END
        return enclosingIntRect(tempRect);
    }
    END_BLOCK_OBJC_EXCEPTIONS
    return IntRect();
}

IntPoint ScrollView::platformScreenToContents(const IntPoint& point) const
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    if (NSView* documentView = this->documentView()) {
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        NSPoint windowCoord = [[documentView window] convertScreenToBase: point];
ALLOW_DEPRECATED_DECLARATIONS_END
        return IntPoint([documentView convertPoint:windowCoord fromView:nil]);
    }
    END_BLOCK_OBJC_EXCEPTIONS
    return IntPoint();
}

bool ScrollView::platformIsOffscreen() const
{
    return ![platformWidget() window] || ![[platformWidget() window] isVisible];
}

static inline NSScrollerKnobStyle toNSScrollerKnobStyle(ScrollbarOverlayStyle style)
{
    switch (style) {
    case ScrollbarOverlayStyleDark:
        return NSScrollerKnobStyleDark;
    case ScrollbarOverlayStyleLight:
        return NSScrollerKnobStyleLight;
    default:
        return NSScrollerKnobStyleDefault;
    }
}

void ScrollView::platformSetScrollbarOverlayStyle(ScrollbarOverlayStyle overlayStyle)
{
    [scrollView() setScrollerKnobStyle:toNSScrollerKnobStyle(overlayStyle)];
}

void ScrollView::platformSetScrollOrigin(const IntPoint& origin, bool updatePositionAtAll, bool updatePositionSynchronously)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [scrollView() setScrollOrigin:origin updatePositionAtAll:updatePositionAtAll immediately:updatePositionSynchronously];
    END_BLOCK_OBJC_EXCEPTIONS
}

} // namespace WebCore

#endif // PLATFORM(MAC)
