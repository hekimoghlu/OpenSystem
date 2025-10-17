/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
#import <WebCore/ScrollTypes.h>

#if PLATFORM(IOS_FAMILY)

namespace WebCore {
class FloatPoint;
}


@class WKVelocityTrackingScrollView;
@class WebEvent;

@protocol WKKeyboardScrollViewAnimatorDelegate;

@interface WKKeyboardScrollViewAnimator : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithScrollView:(WKVelocityTrackingScrollView *)scrollView;

- (void)invalidate;

- (void)willStartInteractiveScroll;

- (BOOL)beginWithEvent:(::WebEvent *)event;
- (void)handleKeyEvent:(::WebEvent *)event;

- (BOOL)scrollTriggeringKeyIsPressed;

- (void)stopScrollingImmediately;

@property (nonatomic, weak) id <WKKeyboardScrollViewAnimatorDelegate> delegate;

@end

@protocol WKKeyboardScrollViewAnimatorDelegate <NSObject>
@optional
- (BOOL)isScrollableForKeyboardScrollViewAnimator:(WKKeyboardScrollViewAnimator *)animator;
- (CGFloat)keyboardScrollViewAnimator:(WKKeyboardScrollViewAnimator *)animator distanceForIncrement:(WebCore::ScrollGranularity)increment inDirection:(WebCore::ScrollDirection)direction;
- (void)keyboardScrollViewAnimatorWillScroll:(WKKeyboardScrollViewAnimator *)animator;
- (void)keyboardScrollViewAnimatorDidFinishScrolling:(WKKeyboardScrollViewAnimator *)animator;

@end

#endif // PLATFORM(IOS_FAMILY)
