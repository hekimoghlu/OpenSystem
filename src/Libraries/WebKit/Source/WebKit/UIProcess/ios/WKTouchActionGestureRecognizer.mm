/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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
#import "WKTouchActionGestureRecognizer.h"

#if PLATFORM(IOS_FAMILY)

#import <UIKit/UIGestureRecognizerSubclass.h>
#import <wtf/HashMap.h>

@implementation WKTouchActionGestureRecognizer {
    HashMap<unsigned, OptionSet<WebCore::TouchAction>> _touchActionsByTouchIdentifier;
    id <WKTouchActionGestureRecognizerDelegate> _touchActionDelegate;
}

- (id)initWithTouchActionDelegate:(id <WKTouchActionGestureRecognizerDelegate>)touchActionDelegate
{
    if (self = [super init])
        _touchActionDelegate = touchActionDelegate;
    return self;
}

- (void)setTouchActions:(OptionSet<WebCore::TouchAction>)touchActions forTouchIdentifier:(unsigned)touchIdentifier
{
    ASSERT(!touchActions.contains(WebCore::TouchAction::Auto));
    _touchActionsByTouchIdentifier.set(touchIdentifier, touchActions);
}

- (void)clearTouchActionsForTouchIdentifier:(unsigned)touchIdentifier
{
    _touchActionsByTouchIdentifier.remove(touchIdentifier);
}

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [self _updateState];
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [self _updateState];
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [self _updateState];
}

- (void)touchesCancelled:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [self _updateState];
}

- (void)_updateState
{
    // We always want to be in a recognized state so that we may always prevent another gesture recognizer.
    [self setState:UIGestureRecognizerStateRecognized];
}

- (BOOL)canBePreventedByGestureRecognizer:(UIGestureRecognizer *)preventingGestureRecognizer
{
    // This allows this gesture recognizer to persist, even if other gesture recognizers are recognized.
    return NO;
}

- (BOOL)canPreventGestureRecognizer:(UIGestureRecognizer *)preventedGestureRecognizer
{
    if (_touchActionsByTouchIdentifier.isEmpty())
        return NO;

    auto mayPan = [_touchActionDelegate gestureRecognizerMayPanWebView:preventedGestureRecognizer];
    auto mayPinchToZoom = [_touchActionDelegate gestureRecognizerMayPinchToZoomWebView:preventedGestureRecognizer];
    auto mayDoubleTapToZoom = [_touchActionDelegate gestureRecognizerMayDoubleTapToZoomWebView:preventedGestureRecognizer];

    if (!mayPan && !mayPinchToZoom && !mayDoubleTapToZoom)
        return NO;

    // Now that we've established that this gesture recognizer may yield an interaction that is preventable by the "touch-action"
    // CSS property we iterate over all active touches, check whether that touch matches the gesture recognizer, see if we have
    // any touch-action specified for it, and then check for each type of interaction whether the touch-action property has a
    // value that should prevent the interaction.
    auto* activeTouches = [_touchActionDelegate touchActionActiveTouches];
    for (NSNumber *touchIdentifier in activeTouches) {
        auto iterator = _touchActionsByTouchIdentifier.find([touchIdentifier unsignedIntegerValue]);
        if (iterator != _touchActionsByTouchIdentifier.end() && [[activeTouches objectForKey:touchIdentifier].gestureRecognizers containsObject:preventedGestureRecognizer]) {
            // Panning is only allowed if "pan-x", "pan-y" or "manipulation" is specified. Additional work is needed to respect individual values, but this takes
            // care of the case where no panning is allowed.
            if (mayPan && !iterator->value.containsAny({ WebCore::TouchAction::PanX, WebCore::TouchAction::PanY, WebCore::TouchAction::Manipulation }))
                return YES;
            // Pinch-to-zoom is only allowed if "pinch-zoom" or "manipulation" is specified.
            if (mayPinchToZoom && !iterator->value.containsAny({ WebCore::TouchAction::PinchZoom, WebCore::TouchAction::Manipulation }))
                return YES;
            // Double-tap-to-zoom is only disallowed if "none" is specified.
            if (mayDoubleTapToZoom && iterator->value.contains(WebCore::TouchAction::None))
                return YES;
        }
    }

    return NO;
}

@end

#endif
