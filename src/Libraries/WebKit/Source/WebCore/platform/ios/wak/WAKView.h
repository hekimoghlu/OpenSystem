/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#ifndef WAKView_h
#define WAKView_h

#import <Foundation/Foundation.h>

#if TARGET_OS_IPHONE

#import <CoreGraphics/CoreGraphics.h>
#import <WebCore/WAKAppKitStubs.h>
#import <WebCore/WAKResponder.h>

extern NSString *WAKViewFrameSizeDidChangeNotification;
extern NSString *WAKViewDidScrollNotification;

#if WAK_APPKIT_API_AVAILABLE_MACCATALYST
#import <AppKit/NSView.h>
#else
enum {
    NSViewNotSizable = 0,
    NSViewMinXMargin = 1,
    NSViewWidthSizable = 2,
    NSViewMaxXMargin = 4,
    NSViewMinYMargin = 8,
    NSViewHeightSizable = 16,
    NSViewMaxYMargin = 32
};
#endif

@class WAKWindow;

WEBCORE_EXPORT @interface WAKView : WAKResponder

+ (WAKView *)focusView;

- (id)initWithFrame:(CGRect)rect;

- (WAKWindow *)window;

- (NSRect)bounds;
- (NSRect)frame;

- (void)setFrame:(NSRect)frameRect;
- (void)setFrameOrigin:(NSPoint)newOrigin;
- (void)setFrameSize:(NSSize)newSize;
- (void)setBoundsOrigin:(NSPoint)newOrigin;
- (void)setBoundsSize:(NSSize)size;
- (void)frameSizeChanged;

- (NSArray *)subviews;
- (WAKView *)superview;
- (void)addSubview:(WAKView *)subview;
- (void)willRemoveSubview:(WAKView *)subview;
- (void)removeFromSuperview;
- (BOOL)isDescendantOf:(WAKView *)aView;
- (BOOL)isHiddenOrHasHiddenAncestor;
- (WAKView *)lastScrollableAncestor;

- (void)viewDidMoveToWindow;

- (void)lockFocus;
- (void)unlockFocus;

- (void)setNeedsDisplay:(BOOL)flag;
- (void)setNeedsDisplayInRect:(CGRect)invalidRect;
- (BOOL)needsDisplay;
- (void)display;
- (void)displayIfNeeded;
- (void)displayRect:(NSRect)rect;
- (void)displayRectIgnoringOpacity:(NSRect)rect;
- (void)displayRectIgnoringOpacity:(NSRect)rect inContext:(CGContextRef)context;
- (void)drawRect:(CGRect)rect;
- (void)viewWillDraw;

- (WAKView *)hitTest:(NSPoint)point;
- (NSPoint)convertPoint:(NSPoint)point fromView:(WAKView *)aView;
- (NSPoint)convertPoint:(NSPoint)point toView:(WAKView *)aView;
- (NSSize)convertSize:(NSSize)size toView:(WAKView *)aView;
- (NSRect)convertRect:(NSRect)rect fromView:(WAKView *)aView;
- (NSRect)convertRect:(NSRect)rect toView:(WAKView *)aView;

- (BOOL)needsPanelToBecomeKey;

- (BOOL)scrollRectToVisible:(NSRect)aRect;
- (void)scrollPoint:(NSPoint)aPoint;
- (NSRect)visibleRect;

- (void)setHidden:(BOOL)flag;

- (void)setNextKeyView:(WAKView *)aView;
- (WAKView *)nextKeyView;
- (WAKView *)nextValidKeyView;
- (WAKView *)previousKeyView;
- (WAKView *)previousValidKeyView;

- (void)invalidateGState;
- (void)releaseGState;

- (void)setAutoresizingMask:(unsigned int)mask;
- (unsigned int)autoresizingMask;
- (BOOL)inLiveResize;

- (BOOL)mouse:(NSPoint)aPoint inRect:(NSRect)aRect;

- (void)setNeedsLayout:(BOOL)flag;
- (void)layout;
- (void)layoutIfNeeded;

- (void)setScale:(float)scale;
- (float)scale;

- (void)_setDrawsOwnDescendants:(BOOL)draw;

- (void)_appendDescriptionToString:(NSMutableString *)info atLevel:(int)level;

+ (void)_setInterpolationQuality:(int)quality;

@end

#endif // TARGET_OS_IPHONE

#endif // WAKView_h
