/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#import "Widget.h"

#if PLATFORM(IOS_FAMILY)

#import "Cursor.h"
#import "Document.h"
#import "FontCascade.h"
#import "GraphicsContext.h"
#import "HostWindow.h"
#import "LocalFrame.h"
#import "LocalFrameView.h"
#import "PlatformMouseEvent.h"
#import "ScrollView.h"
#import "WAKScrollView.h"
#import "WAKView.h"
#import "WAKWindow.h"
#import "WebCoreFrameView.h"
#import "WebCoreView.h"
#import <wtf/BlockObjCExceptions.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

@interface NSView (WebSetSelectedMethods)
- (void)setIsSelected:(BOOL)isSelected;
- (void)webPlugInSetIsSelected:(BOOL)isSelected;
@end

namespace WebCore {

static void safeRemoveFromSuperview(NSView *view)
{
    // FIXME: we should probably change the first responder of the window if
    // it is a descendant of the view we remove.
    // See: <rdar://problem/10360186>
    [view removeFromSuperview];
}

Widget::Widget(NSView* view)
{
    init(view);
}

Widget::~Widget()
{
}

// FIXME: Should move this to Chrome; bad layering that this knows about Frame.
void Widget::setFocus(bool focused)
{
    UNUSED_PARAM(focused);
}

void Widget::show()
{
    if (isSelfVisible())
        return;

    setSelfVisible(true);

    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [getOuterView() setHidden:NO];
    END_BLOCK_OBJC_EXCEPTIONS
}

void Widget::hide()
{
    if (!isSelfVisible())
        return;

    setSelfVisible(false);

    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [getOuterView() setHidden:YES];
    END_BLOCK_OBJC_EXCEPTIONS
}

IntRect Widget::frameRect() const
{
    if (!platformWidget())
        return m_frame;
    
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    return enclosingIntRect([getOuterView() frame]);
    END_BLOCK_OBJC_EXCEPTIONS
    
    return m_frame;
}

void Widget::setFrameRect(const IntRect &rect)
{
    m_frame = rect;
    
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    NSView *v = getOuterView();
    NSRect f = rect;
    if (!NSEqualRects(f, [v frame])) {
        [v setFrame:f];
        [v setNeedsDisplay: NO];
    }
    END_BLOCK_OBJC_EXCEPTIONS
}

NSView* Widget::getOuterView() const
{
    NSView* view = platformWidget();

    // If this widget's view is a WebCoreFrameScrollView then we
    // resize its containing view, a WebFrameView.
    if ([view conformsToProtocol:@protocol(WebCoreFrameScrollView)]) {
        view = [view superview];
        ASSERT(view);
    }

    return view;
}

void Widget::paint(GraphicsContext& p, const IntRect& r, SecurityOriginPaintPolicy, RegionContext*)
{
    if (p.paintingDisabled())
        return;
    
    NSView *view = getOuterView();

    CGContextRef cgContext = p.platformContext();
    CGContextSaveGState(cgContext);

    NSRect viewFrame = [view frame];
    NSRect viewBounds = [view bounds];
    CGContextTranslateCTM(cgContext, viewFrame.origin.x - viewBounds.origin.x, viewFrame.origin.y - viewBounds.origin.y);

    BEGIN_BLOCK_OBJC_EXCEPTIONS
    [view displayRectIgnoringOpacity:[view convertRect:r fromView:[view superview]] inContext:cgContext];
    END_BLOCK_OBJC_EXCEPTIONS

    CGContextRestoreGState(cgContext);
}

void Widget::setIsSelected(bool /*isSelected*/)
{
}

void Widget::addToSuperview(NSView *view)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS

    ASSERT(view);
    NSView *subview = getOuterView();

    if (!subview)
        return;

    ASSERT(![view isDescendantOf:subview]);
    
    if ([subview superview] != view)
        [view addSubview:subview];

    END_BLOCK_OBJC_EXCEPTIONS
}

void Widget::removeFromSuperview()
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    safeRemoveFromSuperview(getOuterView());
    END_BLOCK_OBJC_EXCEPTIONS
}

IntRect Widget::convertFromRootToContainingWindow(const Widget* rootWidget, const IntRect& rect)
{
    if (!rootWidget->platformWidget())
        return rect;

    BEGIN_BLOCK_OBJC_EXCEPTIONS
    WAKScrollView *view = checked_objc_cast<WAKScrollView>(rootWidget->platformWidget());
    if (WAKView *documentView = [view documentView])
        return enclosingIntRect([documentView convertRect:rect toView:nil]);
    return enclosingIntRect([view convertRect:rect toView:nil]);
    END_BLOCK_OBJC_EXCEPTIONS

    return rect;
}

IntRect Widget::convertFromContainingWindowToRoot(const Widget* rootWidget, const IntRect& rect)
{
    if (!rootWidget->platformWidget())
        return rect;

    BEGIN_BLOCK_OBJC_EXCEPTIONS
    WAKScrollView *view = checked_objc_cast<WAKScrollView>(rootWidget->platformWidget());
    if (WAKView *documentView = [view documentView])
        return enclosingIntRect([documentView convertRect:rect fromView:nil]);
    return enclosingIntRect([view convertRect:rect fromView:nil]);
    END_BLOCK_OBJC_EXCEPTIONS

    return rect;
}

IntPoint Widget::convertFromRootToContainingWindow(const Widget* rootWidget, const IntPoint& point)
{
    if (!rootWidget->platformWidget())
        return point;

    BEGIN_BLOCK_OBJC_EXCEPTIONS
    WAKScrollView *view = checked_objc_cast<WAKScrollView>(rootWidget->platformWidget());
    NSPoint convertedPoint;
    if (WAKView *documentView = [view documentView])
        convertedPoint = [documentView convertPoint:point toView:nil];
    else
        convertedPoint = [view convertPoint:point toView:nil];
    return IntPoint(roundf(convertedPoint.x), roundf(convertedPoint.y));
    END_BLOCK_OBJC_EXCEPTIONS

    return point;
}

IntPoint Widget::convertFromContainingWindowToRoot(const Widget* rootWidget, const IntPoint& point)
{
    if (!rootWidget->platformWidget())
        return point;

    BEGIN_BLOCK_OBJC_EXCEPTIONS
    WAKScrollView *view = checked_objc_cast<WAKScrollView>(rootWidget->platformWidget());
    NSPoint convertedPoint;
    if (WAKView *documentView = [view documentView])
        convertedPoint = IntPoint([documentView convertPoint:point fromView:nil]);
    else
        convertedPoint = IntPoint([view convertPoint:point fromView:nil]);
    return IntPoint(roundf(convertedPoint.x), roundf(convertedPoint.y));
    END_BLOCK_OBJC_EXCEPTIONS

    return point;
}

NSView *Widget::platformWidget() const
{
    return m_widget.get();
}

void Widget::setPlatformWidget(NSView *widget)
{
    if (widget == m_widget)
        return;

    m_widget = widget;
}

}

#endif // PLATFORM(IOS_FAMILY)
