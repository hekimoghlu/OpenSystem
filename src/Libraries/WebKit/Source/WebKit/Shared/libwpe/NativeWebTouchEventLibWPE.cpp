/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#include "NativeWebTouchEvent.h"

#if ENABLE(TOUCH_EVENTS)

#include "WebEventFactory.h"

namespace WebKit {

NativeWebTouchEvent::NativeWebTouchEvent(struct wpe_input_touch_event* event, float deviceScaleFactor)
    : WebTouchEvent(WebEventFactory::createWebTouchEvent(event, deviceScaleFactor))
    , m_fallbackTouchPoint { wpe_input_touch_event_type_null, 0, 0, 0, 0 }
{
    for (unsigned i = 0; i < event->touchpoints_length; ++i) {
        WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // WPE port
        auto& point = event->touchpoints[i];
        WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        if (point.type != wpe_input_touch_event_type_null) {
            m_fallbackTouchPoint = point;
            break;
        }
    }
}

} // namespace WebKit

#endif // PLATFORM(TOUCH_EVENTS)
