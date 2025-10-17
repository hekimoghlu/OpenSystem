/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#import "WKInspectorNodeSearchGestureRecognizer.h"

#if PLATFORM(IOS_FAMILY)

#import <UIKit/UIGestureRecognizerSubclass.h>
#import <wtf/RetainPtr.h>

@implementation WKInspectorNodeSearchGestureRecognizer {
    RetainPtr<UITouch> _touch;
}

- (CGPoint)locationInView:(UIView *)view
{
    return [_touch locationInView:view];
}

- (void)_processTouches:(NSSet *)touches state:(UIGestureRecognizerState)newState
{
    ASSERT(_touch);
    if ([touches containsObject:_touch.get()])
        self.state = newState;
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
    if (self.state != UIGestureRecognizerStatePossible)
        return;

    ASSERT(!_touch);
    _touch = [touches anyObject];

    [self _processTouches:touches state:UIGestureRecognizerStateBegan];
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event
{
    [self _processTouches:touches state:UIGestureRecognizerStateChanged];
}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event
{
    [self _processTouches:touches state:UIGestureRecognizerStateEnded];
}

- (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event
{
    [self _processTouches:touches state:UIGestureRecognizerStateCancelled];
}

- (void)reset
{
    _touch = nil;
}

@end

#endif
