/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "WKVelocityTrackingScrollView.h"

@class WKWebView;

@interface WKScrollView : WKVelocityTrackingScrollView

@property (nonatomic, assign) WKWebView <WKBEScrollViewDelegate> *internalDelegate;

- (void)_setBackgroundColorInternal:(UIColor *)backgroundColor;
- (void)_setIndicatorStyleInternal:(UIScrollViewIndicatorStyle)indicatorStyle;
- (void)_setContentSizePreservingContentOffsetDuringRubberband:(CGSize)contentSize;
- (void)_setScrollEnabledInternal:(BOOL)enabled;
- (void)_setZoomEnabledInternal:(BOOL)enabled;
- (void)_setBouncesInternal:(BOOL)horizontal vertical:(BOOL)vertical;
- (BOOL)_setContentScrollInsetInternal:(UIEdgeInsets)insets;
- (void)_setDecelerationRateInternal:(UIScrollViewDecelerationRate)rate;

- (void)_resetContentInset;
@property (nonatomic, readonly) BOOL _contentInsetWasExternallyOverridden;

// FIXME: Likely we can remove this special case for watchOS.
#if !PLATFORM(WATCHOS)
@property (nonatomic, assign, readonly) BOOL _contentInsetAdjustmentBehaviorWasExternallyOverridden;
- (void)_setContentInsetAdjustmentBehaviorInternal:(UIScrollViewContentInsetAdjustmentBehavior)insetAdjustmentBehavior;
- (void)_resetContentInsetAdjustmentBehavior;
#endif

@end

#endif // PLATFORM(IOS_FAMILY)
