/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#import <WebKitLegacy/WebNSViewExtras.h>

#import <WebKitLegacy/DOMExtensions.h>
#import <WebKitLegacy/WebDataSource.h>
#import <WebKitLegacy/WebFramePrivate.h>
#import <WebKitLegacy/WebFrameViewInternal.h>
#import <WebKitLegacy/WebNSImageExtras.h>
#import <WebKitLegacy/WebNSURLExtras.h>
#import <WebKitLegacy/WebView.h>

#if !PLATFORM(IOS_FAMILY)
#import <WebKitLegacy/WebNSPasteboardExtras.h>
#endif

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WAKWindow.h>
#endif

#define WebDragStartHysteresisX                 5.0f
#define WebDragStartHysteresisY                 5.0f
#define WebMaxDragImageSize                     NSMakeSize(400.0f, 400.0f)
#define WebMaxOriginalImageArea                 (1500.0f * 1500.0f)
#define WebDragIconRightInset                   7.0f
#define WebDragIconBottomInset                  3.0f

@implementation NSView (WebExtras)

- (NSView *)_web_superviewOfClass:(Class)class
{
    NSView *view = [self superview];
    while (view  && ![view isKindOfClass:class])
        view = [view superview];
    return view;
}

- (WebFrameView *)_web_parentWebFrameView
{
    return (WebFrameView *)[self _web_superviewOfClass:[WebFrameView class]];
}

#if !PLATFORM(IOS_FAMILY)
// FIXME: Mail is the only client of _webView, remove this method once no versions of Mail need it.
- (WebView *)_webView
{
    return (WebView *)[self _web_superviewOfClass:[WebView class]];
}

/* Determine whether a mouse down should turn into a drag; started as copy of NSTableView code */
- (BOOL)_web_dragShouldBeginFromMouseDown:(NSEvent *)mouseDownEvent
                           withExpiration:(NSDate *)expiration
                              xHysteresis:(float)xHysteresis
                              yHysteresis:(float)yHysteresis
{
    NSEvent *nextEvent, *firstEvent, *dragEvent, *mouseUp;
    BOOL dragIt;

    if ([mouseDownEvent type] != NSEventTypeLeftMouseDown) {
        return NO;
    }

    nextEvent = nil;
    firstEvent = nil;
    dragEvent = nil;
    mouseUp = nil;
    dragIt = NO;

    while ((nextEvent = [[self window] nextEventMatchingMask:(NSEventMaskLeftMouseUp | NSEventMaskLeftMouseDragged)
                                                   untilDate:expiration
                                                      inMode:NSEventTrackingRunLoopMode
                                                     dequeue:YES]) != nil) {
        if (firstEvent == nil) {
            firstEvent = nextEvent;
        }

        if ([nextEvent type] == NSEventTypeLeftMouseDragged) {
            float deltax = ABS([nextEvent locationInWindow].x - [mouseDownEvent locationInWindow].x);
            float deltay = ABS([nextEvent locationInWindow].y - [mouseDownEvent locationInWindow].y);
            dragEvent = nextEvent;

            if (deltax >= xHysteresis) {
                dragIt = YES;
                break;
            }

            if (deltay >= yHysteresis) {
                dragIt = YES;
                break;
            }
        } else if ([nextEvent type] == NSEventTypeLeftMouseUp) {
            mouseUp = nextEvent;
            break;
        }
    }

    // Since we've been dequeuing the events (If we don't, we'll never see the mouse up...),
    // we need to push some of the events back on.  It makes sense to put the first and last
    // drag events and the mouse up if there was one.
    if (mouseUp != nil) {
        [NSApp postEvent:mouseUp atStart:YES];
    }
    if (dragEvent != nil) {
        [NSApp postEvent:dragEvent atStart:YES];
    }
    if (firstEvent != mouseUp && firstEvent != dragEvent) {
        [NSApp postEvent:firstEvent atStart:YES];
    }

    return dragIt;
}

- (BOOL)_web_dragShouldBeginFromMouseDown:(NSEvent *)mouseDownEvent
                           withExpiration:(NSDate *)expiration
{
    return [self _web_dragShouldBeginFromMouseDown:mouseDownEvent
                                    withExpiration:expiration
                                       xHysteresis:WebDragStartHysteresisX
                                       yHysteresis:WebDragStartHysteresisY];
}


- (NSDragOperation)_web_dragOperationForDraggingInfo:(id <NSDraggingInfo>)sender
{
    if (![NSApp modalWindow] && 
        ![[self window] attachedSheet] &&
        [sender draggingSource] != self &&
        [[sender draggingPasteboard] _web_bestURL]) {

        return NSDragOperationCopy;
    }
    
    return NSDragOperationNone;
}

#endif // !PLATFORM(IOS_FAMILY)

- (BOOL)_web_firstResponderIsSelfOrDescendantView
{
    NSResponder *responder = [[self window] firstResponder];
    return (responder && 
           (responder == self || 
           ([responder isKindOfClass:[NSView class]] && [(NSView *)responder isDescendantOf:self])));
}

@end

#if PLATFORM(IOS_FAMILY)
@implementation NSView (WebDocumentViewExtras)

- (WebFrame *)_frame
{
    WebFrameView *webFrameView = [self _web_parentWebFrameView];
    return [webFrameView webFrame];
}

- (WebView *)_webView
{
    // We used to use the view hierarchy exclusively here, but that won't work
    // right when the first viewDidMoveToSuperview call is done, and this will.
    return [[self _frame] webView];
}

@end
#endif
