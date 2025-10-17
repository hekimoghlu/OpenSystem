/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#import <objc/runtime.h>

#if PLATFORM(MAC)
#import <pal/spi/mac/NSImmediateActionGestureRecognizerSPI.h>
#endif // PLATFORM(MAC)
#import <wtf/SoftLinking.h>

#if ENABLE(REVEAL)

#if USE(APPLE_INTERNAL_SDK)

#if PLATFORM(MAC)
#import <Reveal/RVPresenter.h>
#import <Reveal/Reveal.h>
#endif // PLATFORM(MAC)
#import <RevealCore/RVItem_Private.h>
#import <RevealCore/RVSelection.h>
#import <RevealCore/RevealCore.h>
#else // USE(APPLE_INTERNAL_SDK)

@class DDScannerResult;
@class NSMenuItem;
@protocol RVPresenterHighlightDelegate;

@interface RVItem : NSObject <NSSecureCoding>
- (instancetype)initWithText:(NSString *)text selectedRange:(NSRange)selectedRange NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithDDResult:(DDScannerResult *)result NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithURL:(NSURL *)url rangeInContext:(NSRange)rangeInContext;
@property (readonly, nonatomic) NSRange highlightRange;
@end

@interface RVSelection : NSObject
+ (NSRange)revealRangeAtIndex:(NSUInteger)clickIndex selectedRanges:(NSArray <NSValue *> *)selectedRanges shouldUpdateSelection:(BOOL *)shouldUpdateSelection;
@end

#if PLATFORM(MAC)
@interface RVPresentingContext : NSObject
- (instancetype)initWithPointerLocationInView:(NSPoint)pointerLocationInView inView:(NSView *)view highlightDelegate:(id<RVPresenterHighlightDelegate>)highlightDelegate;
@property (readonly) NSArray <NSValue *> * itemRectsInView;
@end

@protocol RVPresenterHighlightDelegate <NSObject>
@required
- (NSArray <NSValue *> *)revealContext:(RVPresentingContext *)context rectsForItem:(RVItem *)item;
@optional
- (void)revealContext:(RVPresentingContext *)context stopHighlightingItem:(RVItem *)item;
- (void)revealContext:(RVPresentingContext *)context drawRectsForItem:(RVItem *)item;
@end
#endif

@interface RVDocumentContext : NSObject <NSSecureCoding>
@end

@interface RVPresenter : NSObject
#if PLATFORM(MAC)
- (id<NSImmediateActionAnimationController>)animationControllerForItem:(RVItem *)item documentContext:(RVDocumentContext *)documentContext presentingContext:(RVPresentingContext *)presentingContext options:(NSDictionary *)options;
- (BOOL)revealItem:(RVItem *)item documentContext:(RVDocumentContext *)documentContext presentingContext:(RVPresentingContext *)presentingContext options:(NSDictionary *)options;
- (NSArray<NSMenuItem *> *)menuItemsForItem:(RVItem *)item documentContext:(RVDocumentContext *)documentContext presentingContext:(RVPresentingContext *)presentingContext options:(NSDictionary *)options;
#endif // PLATFORM(MAC)
@end

#endif // !USE(APPLE_INTERNAL_SDK)

#endif // ENABLE(REVEAL)
