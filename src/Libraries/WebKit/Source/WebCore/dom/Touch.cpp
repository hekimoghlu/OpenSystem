/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
#include "config.h"

#if ENABLE(TOUCH_EVENTS)

#include "Touch.h"

#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"

namespace WebCore {

static int contentsX(LocalFrame* frame)
{
    if (!frame)
        return 0;
    auto* frameView = frame->view();
    if (!frameView)
        return 0;
    return frameView->scrollX() / frame->pageZoomFactor() / frame->frameScaleFactor();
}

static int contentsY(LocalFrame* frame)
{
    if (!frame)
        return 0;
    auto* frameView = frame->view();
    if (!frameView)
        return 0;
    return frameView->scrollY() / frame->pageZoomFactor() / frame->frameScaleFactor();
}

static LayoutPoint scaledLocation(LocalFrame* frame, int pageX, int pageY)
{
    if (!frame)
        return { pageX, pageY };
    float scaleFactor = frame->pageZoomFactor() * frame->frameScaleFactor();
    return { pageX * scaleFactor, pageY * scaleFactor };
}

Touch::Touch(LocalFrame* frame, EventTarget* target, int identifier, int screenX, int screenY, int pageX, int pageY, int radiusX, int radiusY, float rotationAngle, float force)
    : m_target(target)
    , m_identifier(identifier)
    , m_clientX(pageX - contentsX(frame))
    , m_clientY(pageY - contentsY(frame))
    , m_screenX(screenX)
    , m_screenY(screenY)
    , m_pageX(pageX)
    , m_pageY(pageY)
    , m_radiusX(radiusX)
    , m_radiusY(radiusY)
    , m_rotationAngle(rotationAngle)
    , m_force(force)
    , m_absoluteLocation(scaledLocation(frame, pageX, pageY))
{
}

Touch::Touch(EventTarget* target, int identifier, int clientX, int clientY, int screenX, int screenY, int pageX, int pageY, int radiusX, int radiusY, float rotationAngle, float force, LayoutPoint absoluteLocation)
    : m_target(target)
    , m_identifier(identifier)
    , m_clientX(clientX)
    , m_clientY(clientY)
    , m_screenX(screenX)
    , m_screenY(screenY)
    , m_pageX(pageX)
    , m_pageY(pageY)
    , m_radiusX(radiusX)
    , m_radiusY(radiusY)
    , m_rotationAngle(rotationAngle)
    , m_force(force)
    , m_absoluteLocation(absoluteLocation)
{
}

Ref<Touch> Touch::cloneWithNewTarget(EventTarget* eventTarget) const
{
    return adoptRef(*new Touch(eventTarget, m_identifier, m_clientX, m_clientY, m_screenX, m_screenY, m_pageX, m_pageY, m_radiusX, m_radiusY, m_rotationAngle, m_force, m_absoluteLocation));
}

} // namespace WebCore

#endif
