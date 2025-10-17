/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#if USE(APPKIT)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSScrollerImpPair_Private.h>
#import <AppKit/NSScrollerImp_Private.h>

@interface NSScrollerImp ()
@property (getter=isTracking) BOOL tracking;
@end

@interface NSScrollerImpPair ()
+ (NSUserInterfaceLayoutDirection)scrollerLayoutDirection;
+ (void)_updateAllScrollerImpPairsForNewRecommendedScrollerStyle:(NSScrollerStyle)newRecommendedScrollerStyle;
@end

#else

enum {
    NSOverlayScrollerStateHidden = 0,
    NSOverlayScrollerStateThumbShown = 1,
    NSOverlayScrollerStateAllShown = 2,
    NSOverlayScrollerStatePulseThumb = 3,
};
typedef NSUInteger NSOverlayScrollerState;

@protocol NSScrollerImpDelegate;

@interface NSScrollerImp : NSObject
+ (NSScrollerImp *)scrollerImpWithStyle:(NSScrollerStyle)newScrollerStyle controlSize:(NSControlSize)newControlSize horizontal:(BOOL)horizontal replacingScrollerImp:(id)previous;
@property (retain) CALayer *layer;
- (void)setNeedsDisplay:(BOOL)flag;
@property NSScrollerKnobStyle knobStyle;
@property (getter=isHorizontal) BOOL horizontal;
@property NSSize boundsSize;
@property (getter=isEnabled) BOOL enabled;
@property double doubleValue;
@property double presentationValue;
@property (getter=shouldUsePresentationValue) BOOL usePresentationValue;
@property CGFloat knobProportion;
@property CGFloat uiStateTransitionProgress;
@property CGFloat expansionTransitionProgress;
@property CGFloat trackAlpha;
@property CGFloat knobAlpha;
@property (getter=isExpanded) BOOL expanded;
@property (assign) id<NSScrollerImpDelegate> delegate;
@property (readonly) CGFloat trackBoxWidth;
@property (readonly) CGFloat trackWidth;
@property (readonly) CGFloat trackSideInset;
@property (readonly) CGFloat trackEndInset;
@property (readonly) CGFloat knobEndInset;
@property (readonly) CGFloat knobMinLength;
@property (readonly) CGFloat knobOverlapEndInset;
@property (readonly) CGFloat trackOverlapEndInset;
@property NSUserInterfaceLayoutDirection userInterfaceLayoutDirection;
@property (readonly) NSControlSize controlSize;
- (NSRect)rectForPart:(NSScrollerPart)partCode;
- (void)drawKnobSlotInRect:(NSRect)slotRect highlight:(BOOL)flag alpha:(CGFloat)alpha;
- (void)drawKnobSlotInRect:(NSRect)slotRect highlight:(BOOL)flag;
- (void)drawKnob;
- (void)mouseEnteredScroller;
- (void)mouseExitedScroller;
@end

@interface NSScrollerImp ()
@property (getter=isTracking) BOOL tracking;
@end

@protocol NSScrollerImpDelegate
@required
- (NSRect)convertRectToBacking:(NSRect)aRect;
- (NSRect)convertRectFromBacking:(NSRect)aRect;
- (CALayer *)layer;
- (void)scrollerImp:(NSScrollerImp *)scrollerImp animateKnobAlphaTo:(CGFloat)newKnobAlpha duration:(NSTimeInterval)duration;
- (void)scrollerImp:(NSScrollerImp *)scrollerImp animateTrackAlphaTo:(CGFloat)newTrackAlpha duration:(NSTimeInterval)duration;
- (void)scrollerImp:(NSScrollerImp *)scrollerImp overlayScrollerStateChangedTo:(NSOverlayScrollerState)newOverlayScrollerState;
@optional
- (void)scrollerImp:(NSScrollerImp *)scrollerImp animateUIStateTransitionWithDuration:(NSTimeInterval)duration;
- (void)scrollerImp:(NSScrollerImp *)scrollerImp animateExpansionTransitionWithDuration:(NSTimeInterval)duration;
- (NSPoint)mouseLocationInScrollerForScrollerImp:(NSScrollerImp *)scrollerImp;
- (NSRect)convertRectToLayer:(NSRect)aRect;
- (BOOL)shouldUseLayerPerPartForScrollerImp:(NSScrollerImp *)scrollerImp;
- (NSAppearance *)effectiveAppearanceForScrollerImp:(NSScrollerImp *)scrollerImp;
@end

@protocol NSScrollerImpPairDelegate;

@interface NSScrollerImpPair : NSObject
@property (assign) id<NSScrollerImpPairDelegate> delegate;
@property (retain) NSScrollerImp *verticalScrollerImp;
@property (retain) NSScrollerImp *horizontalScrollerImp;
@property NSScrollerStyle scrollerStyle;
+ (NSUserInterfaceLayoutDirection)scrollerLayoutDirection;
- (void)flashScrollers;
- (void)hideOverlayScrollers;
- (void)lockOverlayScrollerState:(NSOverlayScrollerState)state;
- (void)unlockOverlayScrollerState;
- (BOOL)overlayScrollerStateIsLocked;
- (void)contentAreaScrolled;
- (void)contentAreaScrolledInDirection:(NSPoint)direction;
- (void)contentAreaWillDraw;
- (void)windowOrderedOut;
- (void)windowOrderedIn;
- (void)mouseEnteredContentArea;
- (void)mouseExitedContentArea;
- (void)mouseMovedInContentArea;
- (void)startLiveResize;
- (void)contentAreaDidResize;
- (void)endLiveResize;
- (void)beginScrollGesture;
- (void)endScrollGesture;
+ (void)_updateAllScrollerImpPairsForNewRecommendedScrollerStyle:(NSScrollerStyle)newRecommendedScrollerStyle;
@end

@protocol NSScrollerImpPairDelegate
@required
- (NSRect)contentAreaRectForScrollerImpPair:(NSScrollerImpPair *)scrollerImpPair;
- (BOOL)inLiveResizeForScrollerImpPair:(NSScrollerImpPair *)scrollerImpPair;
- (NSPoint)mouseLocationInContentAreaForScrollerImpPair:(NSScrollerImpPair *)scrollerImpPair;
- (NSPoint)scrollerImpPair:(NSScrollerImpPair *)scrollerImpPair convertContentPoint:(NSPoint)pointInContentArea toScrollerImp:(NSScrollerImp *)scrollerImp;
- (void)scrollerImpPair:(NSScrollerImpPair *)scrollerImpPair setContentAreaNeedsDisplayInRect:(NSRect)rect;
- (void)scrollerImpPair:(NSScrollerImpPair *)scrollerImpPair updateScrollerStyleForNewRecommendedScrollerStyle:(NSScrollerStyle)newRecommendedScrollerStyle;
@optional
- (BOOL)scrollerImpPair:(NSScrollerImpPair *)scrollerImpPair isContentPointVisible:(NSPoint)pointInContentArea;
@end

#endif

WTF_EXTERN_C_BEGIN

NSScrollerStyle _NSRecommendedScrollerStyle();

WTF_EXTERN_C_END

#endif // USE(APPKIT)
