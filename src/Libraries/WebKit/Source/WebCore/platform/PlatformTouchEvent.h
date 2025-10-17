/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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
#ifndef PlatformTouchEvent_h
#define PlatformTouchEvent_h

#include "PlatformEvent.h"
#include "PlatformTouchPoint.h"
#include <wtf/Vector.h>

#if ENABLE(TOUCH_EVENTS)

namespace WebCore {


class PlatformTouchEvent : public PlatformEvent {
public:
    PlatformTouchEvent()
        : PlatformEvent(PlatformEvent::Type::TouchStart)
    {
    }

    const Vector<PlatformTouchPoint>& touchPoints() const { return m_touchPoints; }

    const Vector<PlatformTouchEvent>& coalescedEvents() const { return m_coalescedEvents; }

    const Vector<PlatformTouchEvent>& predictedEvents() const { return m_predictedEvents; }

#if PLATFORM(WPE)
    // FIXME: since WPE currently does not send touch stationary events, we need to be able to set
    // TouchCancelled touchPoints subsequently
    void setTouchPoints(Vector<PlatformTouchPoint>& touchPoints) { m_touchPoints = touchPoints; }
#endif

protected:
    Vector<PlatformTouchPoint> m_touchPoints;
    Vector<PlatformTouchEvent> m_coalescedEvents;
    Vector<PlatformTouchEvent> m_predictedEvents;
};

}

#endif // ENABLE(TOUCH_EVENTS)

#endif // PlatformTouchEvent_h
