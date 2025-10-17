/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#ifndef PlatformTouchPoint_h
#define PlatformTouchPoint_h

#include "IntPoint.h"

#if ENABLE(TOUCH_EVENTS)

namespace WebCore {

class PlatformTouchEvent;

class PlatformTouchPoint {
public:
    enum State {
        TouchReleased,
        TouchPressed,
        TouchMoved,
        TouchStationary,
        TouchCancelled,
        TouchStateEnd // Placeholder: must remain the last item.
    };

    // This is necessary for us to be able to build synthetic events.
    PlatformTouchPoint()
        : m_id(0)
        , m_radiusY(0)
        , m_radiusX(0)
        , m_rotationAngle(0)
        , m_force(0)
    {
    }

#if PLATFORM(WPE)
    // FIXME: since WPE currently does not send touch stationary events, we need to be able to
    // create a PlatformTouchPoint of type TouchCancelled artificially
    PlatformTouchPoint(unsigned id, State state, IntPoint screenPos, IntPoint pos)
        : m_id(id)
        , m_state(state)
        , m_screenPos(screenPos)
        , m_pos(pos)
    {
    }
#endif

    unsigned id() const { return m_id; }
    State state() const { return m_state; }
    IntPoint screenPos() const { return m_screenPos; }
    IntPoint pos() const { return m_pos; }
    int radiusX() const { return m_radiusX; }
    int radiusY() const { return m_radiusY; }
    float rotationAngle() const { return m_rotationAngle; }
    float force() const { return m_force; }

protected:
    unsigned m_id;
    State m_state;
    IntPoint m_screenPos;
    IntPoint m_pos;
    int m_radiusY;
    int m_radiusX;
    float m_rotationAngle;
    float m_force;
};

}

#endif // ENABLE(TOUCH_EVENTS)

#endif // PlatformTouchPoint_h
