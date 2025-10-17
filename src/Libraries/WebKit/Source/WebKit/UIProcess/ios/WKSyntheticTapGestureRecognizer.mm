/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#import "WKSyntheticTapGestureRecognizer.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitUtilities.h"
#import "WKTouchEventsGestureRecognizer.h"
#import <UIKit/UIGestureRecognizerSubclass.h>
#import <wtf/RetainPtr.h>

@implementation WKSyntheticTapGestureRecognizer {
    __weak id _gestureIdentifiedTarget;
    SEL _gestureIdentifiedAction;
    __weak id _gestureFailedTarget;
    SEL _gestureFailedAction;
    __weak id _resetTarget;
    SEL _resetAction;
    RetainPtr<NSNumber> _lastActiveTouchIdentifier;
}

- (void)setGestureIdentifiedTarget:(id)target action:(SEL)action
{
    _gestureIdentifiedTarget = target;
    _gestureIdentifiedAction = action;
}

- (void)setGestureFailedTarget:(id)target action:(SEL)action
{
    _gestureFailedTarget = target;
    _gestureFailedAction = action;
}

- (void)setResetTarget:(id)target action:(SEL)action
{
    _resetTarget = target;
    _resetAction = action;
}

- (void)setState:(UIGestureRecognizerState)state
{
    if (state == UIGestureRecognizerStateEnded)
        [_gestureIdentifiedTarget performSelector:_gestureIdentifiedAction withObject:self];
    else if (state == UIGestureRecognizerStateFailed)
        [_gestureFailedTarget performSelector:_gestureFailedAction withObject:self];
    [super setState:state];
}

- (void)reset
{
    [super reset];

    [_resetTarget performSelector:_resetAction withObject:self];
    _lastActiveTouchIdentifier = nil;
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [super touchesEnded:touches withEvent:event];
    if (!_supportingTouchEventsGestureRecognizer)
        return;

    NSMapTable<NSNumber *, UITouch *> *activeTouches = [_supportingTouchEventsGestureRecognizer activeTouchesByIdentifier];
    for (NSNumber *touchIdentifier in activeTouches) {
        UITouch *touch = [activeTouches objectForKey:touchIdentifier];
        if ([touch.gestureRecognizers containsObject:self]) {
            _lastActiveTouchIdentifier = touchIdentifier;
            break;
        }
    }
}

- (NSNumber*)lastActiveTouchIdentifier
{
    return _lastActiveTouchIdentifier.get();
}

@end

#endif
