/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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

#include "WebEvent.h"
#include <WebCore/IntPoint.h>

namespace WebKit {

#if ENABLE(TOUCH_EVENTS)
#if PLATFORM(IOS_FAMILY)

// FIXME: Having "Platform" in the name makes it sound like this event is platform-specific or
// low-level in some way. That doesn't seem to be the case.
class WebPlatformTouchPoint {
public:
    enum class State : uint8_t {
        Released,
        Pressed,
        Moved,
        Stationary,
        Cancelled
    };

    enum class TouchType : bool {
        Direct,
        Stylus
    };

    WebPlatformTouchPoint() = default;
    WebPlatformTouchPoint(unsigned identifier, WebCore::IntPoint locationInRootView, WebCore::IntPoint locationInViewport, State phase)
        : m_identifier(identifier)
        , m_locationInRootView(locationInRootView)
        , m_locationInViewport(locationInViewport)
        , m_phase(phase)
    {
    }
#if ENABLE(IOS_TOUCH_EVENTS)
    WebPlatformTouchPoint(unsigned identifier, WebCore::IntPoint locationInRootView, WebCore::IntPoint locationInViewport, State phase, double radiusX, double radiusY, double rotationAngle, double force, double altitudeAngle, double azimuthAngle, TouchType touchType)
        : m_identifier(identifier)
        , m_locationInRootView(locationInRootView)
        , m_locationInViewport(locationInViewport)
        , m_phase(phase)
        , m_radiusX(radiusX)
        , m_radiusY(radiusY)
        , m_rotationAngle(rotationAngle)
        , m_force(force)
        , m_altitudeAngle(altitudeAngle)
        , m_azimuthAngle(azimuthAngle)
        , m_touchType(touchType)
    {
    }
#endif

    unsigned identifier() const { return m_identifier; }
    WebCore::IntPoint locationInRootView() const { return m_locationInRootView; }
    WebCore::IntPoint locationInViewport() const { return m_locationInViewport; }
    State phase() const { return m_phase; }
    State state() const { return phase(); }

#if ENABLE(IOS_TOUCH_EVENTS)
    void setRadiusX(double radiusX) { m_radiusX = radiusX; }
    double radiusX() const { return m_radiusX; }
    void setRadiusY(double radiusY) { m_radiusY = radiusY; }
    double radiusY() const { return m_radiusY; }
    void setRotationAngle(double rotationAngle) { m_rotationAngle = rotationAngle; }
    double rotationAngle() const { return m_rotationAngle; }
    void setForce(double force) { m_force = force; }
    double force() const { return m_force; }
    void setAltitudeAngle(double altitudeAngle) { m_altitudeAngle = altitudeAngle; }
    double altitudeAngle() const { return m_altitudeAngle; }
    void setAzimuthAngle(double azimuthAngle) { m_azimuthAngle = azimuthAngle; }
    double azimuthAngle() const { return m_azimuthAngle; }
    void setTouchType(TouchType touchType) { m_touchType = touchType; }
    TouchType touchType() const { return m_touchType; }
#endif


private:
    unsigned m_identifier { 0 };
    WebCore::IntPoint m_locationInRootView;
    WebCore::IntPoint m_locationInViewport;
    State m_phase { State::Released };
#if ENABLE(IOS_TOUCH_EVENTS)
    double m_radiusX { 0 };
    double m_radiusY { 0 };
    double m_rotationAngle { 0 };
    double m_force { 0 };
    double m_altitudeAngle { 0 };
    double m_azimuthAngle { 0 };
    TouchType m_touchType { TouchType::Direct };
#endif
};

class WebTouchEvent : public WebEvent {
public:
    WebTouchEvent(WebEvent&& event, const Vector<WebPlatformTouchPoint>& touchPoints, const Vector<WebTouchEvent>& coalescedEvents, const Vector<WebTouchEvent>& predictedEvents, WebCore::IntPoint position, bool isPotentialTap, bool isGesture, float gestureScale, float gestureRotation, bool canPreventNativeGestures = true)
        : WebEvent(WTFMove(event))
        , m_touchPoints(touchPoints)
        , m_coalescedEvents(coalescedEvents)
        , m_predictedEvents(predictedEvents)
        , m_position(position)
        , m_canPreventNativeGestures(canPreventNativeGestures)
        , m_isPotentialTap(isPotentialTap)
        , m_isGesture(isGesture)
        , m_gestureScale(gestureScale)
        , m_gestureRotation(gestureRotation)
    {
        ASSERT(type() == WebEventType::TouchStart || type() == WebEventType::TouchMove || type() == WebEventType::TouchEnd || type() == WebEventType::TouchCancel);
    }

    const Vector<WebPlatformTouchPoint>& touchPoints() const { return m_touchPoints; }

    const Vector<WebTouchEvent>& coalescedEvents() const { return m_coalescedEvents; }
    void setCoalescedEvents(const Vector<WebTouchEvent>& coalescedEvents) { m_coalescedEvents = coalescedEvents; }

    const Vector<WebTouchEvent>& predictedEvents() const { return m_predictedEvents; }
    void setPredictedEvents(const Vector<WebTouchEvent>& predictedEvents) { m_predictedEvents = predictedEvents; }

    WebCore::IntPoint position() const { return m_position; }
    void setPosition(WebCore::IntPoint position) { m_position = position; }

    bool isPotentialTap() const { return m_isPotentialTap; }

    bool isGesture() const { return m_isGesture; }
    float gestureScale() const { return m_gestureScale; }
    float gestureRotation() const { return m_gestureRotation; }

    bool canPreventNativeGestures() const { return m_canPreventNativeGestures; }
    void setCanPreventNativeGestures(bool canPreventNativeGestures) { m_canPreventNativeGestures = canPreventNativeGestures; }

    bool allTouchPointsAreReleased() const;
    
private:
    Vector<WebPlatformTouchPoint> m_touchPoints;
    Vector<WebTouchEvent> m_coalescedEvents;
    Vector<WebTouchEvent> m_predictedEvents;

    WebCore::IntPoint m_position;
    bool m_canPreventNativeGestures { false };
    bool m_isPotentialTap { false };
    bool m_isGesture { false };
    float m_gestureScale { 0 };
    float m_gestureRotation { 0 };
};

#else // !PLATFORM(IOS_FAMILY)

class WebPlatformTouchPoint {
public:
    enum class State : uint8_t {
        Released,
        Pressed,
        Moved,
        Stationary,
        Cancelled
    };

    WebPlatformTouchPoint()
        : m_rotationAngle(0.0), m_force(0.0) { }

    WebPlatformTouchPoint(uint32_t id, State, const WebCore::IntPoint& screenPosition, const WebCore::IntPoint& position);

    WebPlatformTouchPoint(uint32_t id, State, const WebCore::IntPoint& screenPosition, const WebCore::IntPoint& position, const WebCore::IntSize& radius, float rotationAngle = 0.0, float force = 0.0);
    
    uint32_t id() const { return m_id; }
    State state() const { return m_state; }

    const WebCore::IntPoint& screenPosition() const { return m_screenPosition; }
    const WebCore::IntPoint& position() const { return m_position; }
    const WebCore::IntSize& radius() const { return m_radius; }
    float rotationAngle() const { return m_rotationAngle; }
    float force() const { return m_force; }

    void setState(State state) { m_state = state; }

private:
    uint32_t m_id;
    State m_state;
    WebCore::IntPoint m_screenPosition;
    WebCore::IntPoint m_position;
    WebCore::IntSize m_radius;
    float m_rotationAngle;
    float m_force;
};

class WebTouchEvent : public WebEvent {
public:
    WebTouchEvent(WebEvent&&, Vector<WebPlatformTouchPoint>&&, Vector<WebTouchEvent>&&, Vector<WebTouchEvent>&&);

    const Vector<WebPlatformTouchPoint>& touchPoints() const { return m_touchPoints; }

    const Vector<WebTouchEvent>& coalescedEvents() const { return m_coalescedEvents; }

    const Vector<WebTouchEvent>& predictedEvents() const { return m_predictedEvents; }

    bool allTouchPointsAreReleased() const;

private:
    static bool isTouchEventType(WebEventType);

    Vector<WebPlatformTouchPoint> m_touchPoints;
    Vector<WebTouchEvent> m_coalescedEvents;
    Vector<WebTouchEvent> m_predictedEvents;
};

#endif // PLATFORM(IOS_FAMILY)
#endif // ENABLE(TOUCH_EVENTS)

} // namespace WebKit
