/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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

#import "WKTouchEventsGestureRecognizerTypes.h"
#import <UIKit/UIKit.h>
#import <wtf/Vector.h>

@class WKContentView;

namespace WebKit {

struct WKTouchPoint {
    CGPoint locationInRootViewCoordinates;
    CGPoint locationInViewport;
    unsigned identifier { 0 };
    UITouchPhase phase { UITouchPhaseBegan };
    CGFloat majorRadiusInWindowCoordinates { 0 };
    CGFloat force { 0 };
    CGFloat altitudeAngle { 0 };
    CGFloat azimuthAngle { 0 };
    WKTouchPointType touchType { WKTouchPointType::Direct };
};

struct WKTouchEvent {
    WKTouchEventType type { WKTouchEventType::Begin };
    NSTimeInterval timestamp { 0 };
    CGPoint locationInRootViewCoordinates;
    CGFloat scale { 0 };
    CGFloat rotation { 0 };

    bool inJavaScriptGesture { false };

    Vector<WKTouchPoint> touchPoints;
    Vector<WKTouchEvent> coalescedEvents;
    Vector<WKTouchEvent> predictedEvents;
    bool isPotentialTap { false };
};

} // namespace WebKit

@interface WKTouchEventsGestureRecognizer : UIGestureRecognizer
- (instancetype)initWithContentView:(WKContentView *)view;
- (void)cancel;

@property (nonatomic, getter=isDefaultPrevented) BOOL defaultPrevented;

@property (nonatomic, readonly) const WebKit::WKTouchEvent& lastTouchEvent;
@property (nonatomic, readonly, getter=isDispatchingTouchEvents) BOOL dispatchingTouchEvents;
@property (nonatomic, readonly) NSMapTable<NSNumber *, UITouch *> *activeTouchesByIdentifier;
@property (nonatomic, readonly, weak) WKContentView *contentView;

@end

#endif // PLATFORM(IOS_FAMILY)
