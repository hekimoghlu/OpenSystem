/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
#import <Foundation/Foundation.h>

#if !TARGET_OS_IPHONE
#import <AppKit/AppKit.h>
#else
#import <WebKitLegacy/WAKAppKitStubs.h>
#import <WebKitLegacy/WAKView.h>
#endif

#define WebDragImageAlpha 0.75f

@class DOMElement;
@class WebFrameView;
@class WebView;

#if TARGET_OS_IPHONE
@interface WAKView (WebExtras)
#else
@interface NSView (WebExtras)
#endif

// Returns the nearest enclosing view of the given class, or nil if none.
#if TARGET_OS_IPHONE
- (WAKView *)_web_superviewOfClass:(Class)viewClass;
#else
- (NSView *)_web_superviewOfClass:(Class)viewClass;
#endif
- (WebFrameView *)_web_parentWebFrameView;
#if !TARGET_OS_IPHONE
- (WebView *)_webView;
#endif

#if !TARGET_OS_IPHONE
// returns whether a drag should begin starting with mouseDownEvent; if the time
// passes expiration or the mouse moves less than the hysteresis before the mouseUp event,
// returns NO, else returns YES.
- (BOOL)_web_dragShouldBeginFromMouseDown:(NSEvent *)mouseDownEvent
                           withExpiration:(NSDate *)expiration
                              xHysteresis:(float)xHysteresis
                              yHysteresis:(float)yHysteresis;

// Calls _web_dragShouldBeginFromMouseDown:withExpiration:xHysteresis:yHysteresis: with
// the default values for xHysteresis and yHysteresis
- (BOOL)_web_dragShouldBeginFromMouseDown:(NSEvent *)mouseDownEvent
                           withExpiration:(NSDate *)expiration;

// Convenience method. Returns NSDragOperationCopy if _web_bestURLFromPasteboard doesn't return nil.
// Returns NSDragOperationNone otherwise.
- (NSDragOperation)_web_dragOperationForDraggingInfo:(id <NSDraggingInfo>)sender;

#endif

- (BOOL)_web_firstResponderIsSelfOrDescendantView;

@end

#if TARGET_OS_IPHONE
@class WebFrame;
@class WebView;

@interface WAKView (WebDocumentViewExtras)
- (WebFrame *)_frame;
- (WebView *)_webView;
@end
#endif
