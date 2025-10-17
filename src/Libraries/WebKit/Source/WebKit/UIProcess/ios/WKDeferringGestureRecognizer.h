/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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

#import <UIKit/UIKit.h>

namespace WebKit {

enum class ShouldDeferGestures : bool { No, Yes };
enum class ShouldPreventGestures : bool { No, Yes };

} // namespace WebKit

@class WKDeferringGestureRecognizer;

@protocol WKDeferringGestureRecognizerDelegate
- (BOOL)deferringGestureRecognizer:(WKDeferringGestureRecognizer *)deferringGestureRecognizer shouldDeferOtherGestureRecognizer:(UIGestureRecognizer *)gestureRecognizer;
- (WebKit::ShouldDeferGestures)deferringGestureRecognizer:(WKDeferringGestureRecognizer *)deferringGestureRecognizer willBeginTouchesWithEvent:(UIEvent *)event;
- (void)deferringGestureRecognizer:(WKDeferringGestureRecognizer *)deferringGestureRecognizer didEndTouchesWithEvent:(UIEvent *)event;
- (void)deferringGestureRecognizer:(WKDeferringGestureRecognizer *)deferringGestureRecognizer didTransitionToState:(UIGestureRecognizerState)state;
@end

@interface WKDeferringGestureRecognizer : UIGestureRecognizer

- (instancetype)initWithDeferringGestureDelegate:(id <WKDeferringGestureRecognizerDelegate>)deferringGestureDelegate;

- (BOOL)shouldDeferGestureRecognizer:(UIGestureRecognizer *)gestureRecognizer;
- (void)endDeferral:(WebKit::ShouldPreventGestures)shouldPreventGestures;

@property (nonatomic) BOOL immediatelyFailsAfterTouchEnd;

@end

#endif // PLATFORM(IOS_FAMILY)
