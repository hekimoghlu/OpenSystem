/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#include "WPEGestureDetector.h"
#include "WPEDisplay.h"
#include "WPESettings.h"

#include <cmath>

namespace WPE {

void GestureDetector::handleEvent(WPEEvent* event)
{
    if (m_sequenceId && *m_sequenceId != wpe_event_touch_get_sequence_id(event))
        return;

    switch (wpe_event_get_event_type(event)) {
    case WPE_EVENT_TOUCH_DOWN:
        reset();
        if (double x, y; wpe_event_get_position(event, &x, &y)) {
            m_gesture = WPE_GESTURE_TAP;
            m_position = { x, y };
            m_sequenceId = wpe_event_touch_get_sequence_id(event);
        }
        break;
    case WPE_EVENT_TOUCH_CANCEL:
        reset();
        break;
    case WPE_EVENT_TOUCH_MOVE:
        if (double x, y; wpe_event_get_position(event, &x, &y) && m_position) {
            auto* settings = wpe_display_get_settings(wpe_view_get_display(wpe_event_get_view(event)));
            auto dragActivationThresholdPx = wpe_settings_get_uint32(settings, WPE_SETTING_DRAG_THRESHOLD, nullptr);
            if (m_gesture != WPE_GESTURE_DRAG && std::hypot(x - m_position->x, y - m_position->y) > dragActivationThresholdPx) {
                m_gesture = WPE_GESTURE_DRAG;
                m_nextDeltaReferencePosition = m_position;
                m_dragBegin = true;
            } else if (m_gesture == WPE_GESTURE_DRAG)
                m_dragBegin = false;
            if (m_gesture == WPE_GESTURE_DRAG) {
                m_delta = { x - m_nextDeltaReferencePosition->x, y - m_nextDeltaReferencePosition->y };
                m_nextDeltaReferencePosition = { x, y };
            }
        } else
            reset();
        break;
    case WPE_EVENT_TOUCH_UP:
        if (double x, y; wpe_event_get_position(event, &x, &y) && m_position) {
            if (m_gesture == WPE_GESTURE_DRAG)
                m_delta = { x - m_nextDeltaReferencePosition->x, y - m_nextDeltaReferencePosition->y };
        } else
            reset();
        m_sequenceId = std::nullopt; // We can accept new sequence at this point.
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

void GestureDetector::reset()
{
    m_gesture = WPE_GESTURE_NONE;
    m_sequenceId = std::nullopt;
    m_position = std::nullopt;
    m_nextDeltaReferencePosition = std::nullopt;
    m_delta = std::nullopt;
    m_dragBegin = std::nullopt;
}

} // namespace WPE
