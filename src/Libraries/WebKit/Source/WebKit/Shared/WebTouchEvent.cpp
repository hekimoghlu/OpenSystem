/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
#include "WebTouchEvent.h"

#if ENABLE(TOUCH_EVENTS)

#include "ArgumentCoders.h"

namespace WebKit {

#if !PLATFORM(IOS_FAMILY)

WebTouchEvent::WebTouchEvent(WebEvent&& event, Vector<WebPlatformTouchPoint>&& touchPoints, Vector<WebTouchEvent>&& coalescedEvents, Vector<WebTouchEvent>&& predictedEvents)
    : WebEvent(WTFMove(event))
    , m_touchPoints(WTFMove(touchPoints))
    , m_coalescedEvents(WTFMove(coalescedEvents))
    , m_predictedEvents(WTFMove(predictedEvents))
{
    ASSERT(isTouchEventType(type()));
}

bool WebTouchEvent::isTouchEventType(WebEventType type)
{
    return type == WebEventType::TouchStart || type == WebEventType::TouchMove || type == WebEventType::TouchEnd || type == WebEventType::TouchCancel;
}

#endif // !PLATFORM(IOS_FAMILY)

bool WebTouchEvent::allTouchPointsAreReleased() const
{
    for (const auto& touchPoint : touchPoints()) {
        if (touchPoint.state() != WebPlatformTouchPoint::State::Released && touchPoint.state() != WebPlatformTouchPoint::State::Cancelled)
            return false;
    }

    return true;
}

} // namespace WebKit

#endif // ENABLE(TOUCH_EVENTS)
