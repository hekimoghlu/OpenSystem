/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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

#if ENABLE(IOS_TOUCH_EVENTS)
#include <WebKitAdditions/TouchIOS.h>
#elif ENABLE(TOUCH_EVENTS)

#include "EventTarget.h"
#include "LayoutPoint.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class LocalFrame;

class Touch : public RefCounted<Touch> {
public:
    static Ref<Touch> create(LocalFrame* frame, EventTarget* target,
            int identifier, int screenX, int screenY, int pageX, int pageY,
            int radiusX, int radiusY, float rotationAngle, float force)
    {
        return adoptRef(*new Touch(frame, target, identifier, screenX, 
                screenY, pageX, pageY, radiusX, radiusY, rotationAngle, force));
    }

    EventTarget* target() const { return m_target.get(); }
    int identifier() const { return m_identifier; }
    int clientX() const { return m_clientX; }
    int clientY() const { return m_clientY; }
    int screenX() const { return m_screenX; }
    int screenY() const { return m_screenY; }
    int pageX() const { return m_pageX; }
    int pageY() const { return m_pageY; }
    int webkitRadiusX() const { return m_radiusX; }
    int webkitRadiusY() const { return m_radiusY; }
    float webkitRotationAngle() const { return m_rotationAngle; }
    float webkitForce() const { return m_force; }
    const LayoutPoint& absoluteLocation() const { return m_absoluteLocation; }
    Ref<Touch> cloneWithNewTarget(EventTarget*) const;

private:
    Touch(LocalFrame*, EventTarget*, int identifier,
            int screenX, int screenY, int pageX, int pageY,
            int radiusX, int radiusY, float rotationAngle, float force);

    Touch(EventTarget*, int identifier, int clientX, int clientY,
        int screenX, int screenY, int pageX, int pageY,
        int radiusX, int radiusY, float rotationAngle, float force, LayoutPoint absoluteLocation);

    RefPtr<EventTarget> m_target;
    int m_identifier;
    int m_clientX;
    int m_clientY;
    int m_screenX;
    int m_screenY;
    int m_pageX;
    int m_pageY;
    int m_radiusX;
    int m_radiusY;
    float m_rotationAngle;
    float m_force;
    LayoutPoint m_absoluteLocation;
};

} // namespace WebCore

#endif // ENABLE(TOUCH_EVENTS)
