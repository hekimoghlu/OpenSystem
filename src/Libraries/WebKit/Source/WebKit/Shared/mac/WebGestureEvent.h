/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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

#if ENABLE(MAC_GESTURE_EVENTS)

#include "WebEvent.h"
#include <WebCore/FloatPoint.h>
#include <WebCore/FloatSize.h>
#include <WebCore/IntPoint.h>
#include <WebCore/IntSize.h>
#include <wtf/text/WTFString.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

class WebGestureEvent : public WebEvent {
public:
    WebGestureEvent(WebEvent&& event, WebCore::IntPoint position, float gestureScale, float gestureRotation)
        : WebEvent(WTFMove(event))
        , m_position(position)
        , m_gestureScale(gestureScale)
        , m_gestureRotation(gestureRotation)
    {
        ASSERT(isGestureEventType(type()));
    }

    WebCore::IntPoint position() const { return m_position; }

    float gestureScale() const { return m_gestureScale; }
    float gestureRotation() const { return m_gestureRotation; }
    
private:
    bool isGestureEventType(WebEventType) const;

    WebCore::IntPoint m_position;
    float m_gestureScale;
    float m_gestureRotation;
};

} // namespace WebKit

#endif // ENABLE(MAC_GESTURE_EVENTS)
