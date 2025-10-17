/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#import "WKDeferringGestureRecognizer.h"

#if PLATFORM(IOS_FAMILY)

#import <wtf/WeakObjCPtr.h>

@implementation WKDeferringGestureRecognizer {
    WeakObjCPtr<id <WKDeferringGestureRecognizerDelegate>> _deferringGestureDelegate;
}

- (instancetype)initWithDeferringGestureDelegate:(id <WKDeferringGestureRecognizerDelegate>)deferringGestureDelegate
{
    if (self = [super init])
        _deferringGestureDelegate = deferringGestureDelegate;
    return self;
}

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    auto shouldDeferGestures = [_deferringGestureDelegate deferringGestureRecognizer:self willBeginTouchesWithEvent:event];
    [super touchesBegan:touches withEvent:event];

    if (shouldDeferGestures == WebKit::ShouldDeferGestures::No)
        self.state = UIGestureRecognizerStateFailed;
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [super touchesEnded:touches withEvent:event];

    if (self.immediatelyFailsAfterTouchEnd)
        self.state = UIGestureRecognizerStateFailed;

    [_deferringGestureDelegate deferringGestureRecognizer:self didEndTouchesWithEvent:event];
}

- (void)touchesCancelled:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [super touchesCancelled:touches withEvent:event];
    self.state = UIGestureRecognizerStateFailed;
}

- (void)endDeferral:(WebKit::ShouldPreventGestures)shouldPreventGestures
{
    if (shouldPreventGestures == WebKit::ShouldPreventGestures::Yes)
        self.state = UIGestureRecognizerStateEnded;
    else
        self.state = UIGestureRecognizerStateFailed;
}

- (BOOL)canBePreventedByGestureRecognizer:(UIGestureRecognizer *)preventingGestureRecognizer
{
    return NO;
}

- (BOOL)shouldDeferGestureRecognizer:(UIGestureRecognizer *)gestureRecognizer
{
    return [_deferringGestureDelegate deferringGestureRecognizer:self shouldDeferOtherGestureRecognizer:gestureRecognizer];
}

- (void)setState:(UIGestureRecognizerState)state
{
    auto previousState = self.state;
    [super setState:state];

    if (previousState != self.state)
        [_deferringGestureDelegate deferringGestureRecognizer:self didTransitionToState:state];
}

@end

#endif // PLATFORM(IOS_FAMILY)
