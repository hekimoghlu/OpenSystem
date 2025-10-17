/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
#import "NativeWebTouchEvent.h"

#if PLATFORM(IOS_FAMILY)

#import "WKTouchEventsGestureRecognizer.h"
#import <WebCore/IntPoint.h>
#import <WebCore/WAKAppKitStubs.h>

namespace WebKit {

#if ENABLE(TOUCH_EVENTS)

static inline WebEventType webEventTypeForWKTouchEventType(WKTouchEventType type)
{
    switch (type) {
    case WKTouchEventType::Begin:
        return WebEventType::TouchStart;
    case WKTouchEventType::Change:
        return WebEventType::TouchMove;
    case WKTouchEventType::End:
        return WebEventType::TouchEnd;
    case WKTouchEventType::Cancel:
        return WebEventType::TouchCancel;
    }
}

static WebPlatformTouchPoint::State convertTouchPhase(UITouchPhase touchPhase)
{
    switch (touchPhase) {
    case UITouchPhaseBegan:
        return WebPlatformTouchPoint::State::Pressed;
    case UITouchPhaseMoved:
        return WebPlatformTouchPoint::State::Moved;
    case UITouchPhaseStationary:
        return WebPlatformTouchPoint::State::Stationary;
    case UITouchPhaseEnded:
        return WebPlatformTouchPoint::State::Released;
    case UITouchPhaseCancelled:
        return WebPlatformTouchPoint::State::Cancelled;
    default:
        ASSERT_NOT_REACHED();
        return WebPlatformTouchPoint::State::Stationary;
    }
}

static WebPlatformTouchPoint::TouchType convertTouchType(WKTouchPointType touchType)
{
    switch (touchType) {
    case WKTouchPointType::Direct:
        return WebPlatformTouchPoint::TouchType::Direct;
    case WKTouchPointType::Stylus:
        return WebPlatformTouchPoint::TouchType::Stylus;
    default:
        ASSERT_NOT_REACHED();
        return WebPlatformTouchPoint::TouchType::Direct;
    }
}

static inline WebCore::IntPoint positionForCGPoint(CGPoint position)
{
    return WebCore::IntPoint(position);
}

static CGFloat radiusForTouchPoint(const WKTouchPoint& touchPoint)
{
#if ENABLE(FIXED_IOS_TOUCH_POINT_RADIUS)
    return 12.1;
#else
    return touchPoint.majorRadiusInWindowCoordinates;
#endif
}

Vector<WebPlatformTouchPoint> NativeWebTouchEvent::extractWebTouchPoints(const WKTouchEvent& event)
{
    return event.touchPoints.map([](auto& touchPoint) {
        unsigned identifier = touchPoint.identifier;
        auto locationInRootView = positionForCGPoint(touchPoint.locationInRootViewCoordinates);
        auto locationInViewport = positionForCGPoint(touchPoint.locationInViewport);
        WebPlatformTouchPoint::State phase = convertTouchPhase(touchPoint.phase);
        WebPlatformTouchPoint platformTouchPoint = WebPlatformTouchPoint(identifier, locationInRootView, locationInViewport, phase);
#if ENABLE(IOS_TOUCH_EVENTS)
        auto radius = radiusForTouchPoint(touchPoint);
        platformTouchPoint.setRadiusX(radius);
        platformTouchPoint.setRadiusY(radius);
        // FIXME (259068): Add support for Touch.rotationAngle.
        platformTouchPoint.setRotationAngle(0);
        platformTouchPoint.setForce(touchPoint.force);
        platformTouchPoint.setAltitudeAngle(touchPoint.altitudeAngle);
        platformTouchPoint.setAzimuthAngle(touchPoint.azimuthAngle);
        platformTouchPoint.setTouchType(convertTouchType(touchPoint.touchType));
#endif
        return platformTouchPoint;
    });
}

Vector<WebTouchEvent> NativeWebTouchEvent::extractCoalescedWebTouchEvents(const WKTouchEvent& event, UIKeyModifierFlags flags)
{
    return event.coalescedEvents.map([&](auto& event) -> WebTouchEvent {
        return NativeWebTouchEvent { event, flags };
    });
}

Vector<WebTouchEvent> NativeWebTouchEvent::extractPredictedWebTouchEvents(const WKTouchEvent& event, UIKeyModifierFlags flags)
{
    return event.predictedEvents.map([&](auto& event) -> WebTouchEvent {
        return NativeWebTouchEvent { event, flags };
    });
}

NativeWebTouchEvent::NativeWebTouchEvent(const WKTouchEvent& event, UIKeyModifierFlags flags)
    : WebTouchEvent(
        { webEventTypeForWKTouchEventType(event.type), webEventModifierFlags(flags), WallTime::fromRawSeconds(event.timestamp) },
        extractWebTouchPoints(event),
        extractCoalescedWebTouchEvents(event, flags),
        extractPredictedWebTouchEvents(event, flags),
        positionForCGPoint(event.locationInRootViewCoordinates),
        event.isPotentialTap,
        event.inJavaScriptGesture,
        event.scale,
        event.rotation)
{
}

#endif // ENABLE(TOUCH_EVENTS)

OptionSet<WebEventModifier> webEventModifierFlags(UIKeyModifierFlags flags)
{
    OptionSet<WebEventModifier> modifiers;
    if (flags & UIKeyModifierShift)
        modifiers.add(WebEventModifier::ShiftKey);
    if (flags & UIKeyModifierControl)
        modifiers.add(WebEventModifier::ControlKey);
    if (flags & UIKeyModifierAlternate)
        modifiers.add(WebEventModifier::AltKey);
    if (flags & UIKeyModifierCommand)
        modifiers.add(WebEventModifier::MetaKey);
    if (flags & UIKeyModifierAlphaShift)
        modifiers.add(WebEventModifier::CapsLockKey);
    return modifiers;
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
