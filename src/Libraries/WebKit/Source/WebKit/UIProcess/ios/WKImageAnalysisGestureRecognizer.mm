/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
#import "WKImageAnalysisGestureRecognizer.h"

#if PLATFORM(IOS_FAMILY) && ENABLE(IMAGE_ANALYSIS)

#import "UIKitUtilities.h"

@implementation WKImageAnalysisGestureRecognizer {
    __weak UIView <WKImageAnalysisGestureRecognizerDelegate> *_imageAnalysisGestureRecognizerDelegate;
    __weak UIScrollView *_lastTouchedScrollView;
}

- (instancetype)initWithImageAnalysisGestureDelegate:(UIView <WKImageAnalysisGestureRecognizerDelegate> *)delegate
{
    if (!(self = [super init]))
        return nil;

    _imageAnalysisGestureRecognizerDelegate = delegate;
    self.delegate = delegate;
    self.minimumPressDuration = 0.1;
    self.allowableMovement = 0;
    self.name = @"Image analysis";
    return self;
}

- (void)reset
{
    [super reset];

    _lastTouchedScrollView = nil;
}

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [super touchesBegan:touches withEvent:event];

    if (auto scrollView = WebKit::scrollViewForTouches(touches))
        _lastTouchedScrollView = scrollView;

    [self beginAfterExceedingForceThresholdIfNeeded:touches];
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [super touchesMoved:touches withEvent:event];

    [self beginAfterExceedingForceThresholdIfNeeded:touches];
}

- (void)beginAfterExceedingForceThresholdIfNeeded:(NSSet<UITouch *> *)touches
{
    if (self.state != UIGestureRecognizerStatePossible)
        return;

    if (touches.count > 1)
        return;

    constexpr CGFloat forceThreshold = 1.5;
    if (touches.anyObject.force < forceThreshold)
        return;

    self.state = UIGestureRecognizerStateBegan;
}

- (void)setState:(UIGestureRecognizerState)state
{
    auto previousState = self.state;
    super.state = state;

    auto newState = self.state;
    if (previousState == newState)
        return;

    if (newState == UIGestureRecognizerStateBegan)
        [_imageAnalysisGestureRecognizerDelegate imageAnalysisGestureDidBegin:self];
    else if (newState == UIGestureRecognizerStateFailed)
        [_imageAnalysisGestureRecognizerDelegate imageAnalysisGestureDidFail:self];
}

@end

#endif // PLATFORM(IOS_FAMILY) && ENABLE(IMAGE_ANALYSIS)
