/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#pragma once

#if PLATFORM(IOS_FAMILY)

#import <UIKit/UIKit.h>
#import <wtf/RetainPtr.h>

@interface UIScrollView (WebKitInternal)
@property (readonly, nonatomic) BOOL _wk_isInterruptingDeceleration;
@property (readonly, nonatomic) BOOL _wk_isScrolledBeyondExtents;
@property (readonly, nonatomic) BOOL _wk_canScrollHorizontallyWithoutBouncing;
@property (readonly, nonatomic) BOOL _wk_canScrollVerticallyWithoutBouncing;
@property (readonly, nonatomic) CGFloat _wk_contentWidthIncludingInsets;
@property (readonly, nonatomic) CGFloat _wk_contentHeightIncludingInsets;
@property (readonly, nonatomic) BOOL _wk_isScrollAnimating;
@property (readonly, nonatomic) BOOL _wk_isZoomAnimating;
- (void)_wk_setContentOffsetAndShowScrollIndicators:(CGPoint)offset animated:(BOOL)animated;
- (void)_wk_setTransfersHorizontalScrollingToParent:(BOOL)value;
- (void)_wk_setTransfersVerticalScrollingToParent:(BOOL)value;
- (void)_wk_stopScrollingAndZooming;
- (CGPoint)_wk_clampToScrollExtents:(CGPoint)contentOffset;
@end

@interface UIGestureRecognizer (WebKitInternal)
@property (nonatomic, readonly) BOOL _wk_isTextInteractionLoupeGesture;
@property (nonatomic, readonly) BOOL _wk_isTextInteractionTapGesture;
@property (nonatomic, readonly) BOOL _wk_hasRecognizedOrEnded;
@end

@interface UIView (WebKitInternal)
- (BOOL)_wk_isAncestorOf:(UIView *)view;
@property (nonatomic, readonly) UIScrollView *_wk_parentScrollView;
@property (nonatomic, readonly) UIViewController *_wk_viewControllerForFullScreenPresentation;
@end

@interface UIViewController (WebKitInternal)
@property (nonatomic, readonly) BOOL _wk_isInFullscreenPresentation;
@end

namespace WebKit {

RetainPtr<UIAlertController> createUIAlertController(NSString *title, NSString *message);
UIScrollView *scrollViewForTouches(NSSet<UITouch *> *);

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
