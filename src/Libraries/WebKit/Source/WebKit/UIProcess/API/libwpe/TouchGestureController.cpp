/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
#include "TouchGestureController.h"

#if USE(LIBWPE) && ENABLE(TOUCH_EVENTS)

#include <WebCore/Scrollbar.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

// FIXME: These ought to be either configurable or derived from system
//        properties, such as screen size and pixel density.
static constexpr uint32_t scrollCaptureThreshold { 200 };
static constexpr uint32_t axisLockMovementThreshold { 8 };
static constexpr uint32_t axisLockActivationThreshold { 15 };
static constexpr uint32_t axisLockReleaseThreshold { 30 };
static constexpr uint32_t contextMenuThreshold { 500 };

WTF_MAKE_TZONE_ALLOCATED_IMPL(TouchGestureController);

TouchGestureController::EventVariant TouchGestureController::handleEvent(const struct wpe_input_touch_event_raw* touchPoint)
{
    switch (touchPoint->type) {
    case wpe_input_touch_event_type_down:
        // Start of the touch interaction, first possible event is a mouse click.
        m_gesturedEvent = GesturedEvent::Click;
        m_start = { true, touchPoint->time, touchPoint->x, touchPoint->y };
        m_offset = { touchPoint->x, touchPoint->y };
        m_xAxisLockBroken = m_yAxisLockBroken = false;
        break;
    case wpe_input_touch_event_type_motion:
    {
        switch (m_gesturedEvent) {
        case GesturedEvent::None:
            break;
        case GesturedEvent::Click:
        {
            // If currently only gesturing a click, determine if the touch has progressed
            // so far that it should become a scrolling gesture.
            int32_t deltaX = touchPoint->x - m_start.x;
            int32_t deltaY = touchPoint->y - m_start.y;
            uint32_t deltaTime = touchPoint->time - m_start.time;

            int pixelsPerLineStep = WebCore::Scrollbar::pixelsPerLineStep();
            bool overThreshold = std::abs(deltaX) >= pixelsPerLineStep
                || std::abs(deltaY) >= pixelsPerLineStep
                || deltaTime >= scrollCaptureThreshold;
            if (!overThreshold)
                break;

            // Over threshold, bump the gestured event and directly fall through to handling it.
            m_gesturedEvent = GesturedEvent::Axis;
            FALLTHROUGH;
        }
        case GesturedEvent::ContextMenu:
            break;
        case GesturedEvent::Axis:
        {
            AxisEvent generatedEvent;
            generatedEvent.phase = WebWheelEvent::Phase::PhaseChanged;

#if WPE_CHECK_VERSION(1, 5, 0)
            generatedEvent.event.base = {
                static_cast<enum wpe_input_axis_event_type>(wpe_input_axis_event_type_mask_2d | wpe_input_axis_event_type_motion_smooth),
                touchPoint->time, m_start.x, m_start.y,
                0, 0, 0,
            };

            uint32_t xOffset = std::abs(touchPoint->x - m_start.x);
            uint32_t yOffset = std::abs(touchPoint->y - m_start.y);

            if (xOffset >= axisLockReleaseThreshold)
                m_xAxisLockBroken = true;
            if (yOffset >= axisLockReleaseThreshold)
                m_yAxisLockBroken = true;

            if (xOffset >= axisLockMovementThreshold && yOffset >= axisLockMovementThreshold && xOffset < axisLockActivationThreshold && yOffset < axisLockActivationThreshold) {
                m_xAxisLockBroken = true;
                m_yAxisLockBroken = true;
            }

            generatedEvent.event.x_axis = (m_xAxisLockBroken || yOffset < axisLockActivationThreshold) ?  -(m_offset.x - touchPoint->x) : 0;
            generatedEvent.event.y_axis = (m_yAxisLockBroken || xOffset < axisLockActivationThreshold) ?  -(m_offset.y - touchPoint->y) : 0;
#else
            generatedEvent.event = {
                wpe_input_axis_event_type_motion,
                touchPoint->time, m_start.x, m_start.y,
                2, (touchPoint->y - m_offset.y), 0
            };
#endif
            m_offset.x = touchPoint->x;
            m_offset.y = touchPoint->y;
            return generatedEvent;
        }
        }
        break;
    }
    case wpe_input_touch_event_type_up:
    {
        switch (m_gesturedEvent) {
        case GesturedEvent::None:
            break;
        case GesturedEvent::Click:
        {
            bool generateClick = true;

#if ENABLE(CONTEXT_MENUS)
            generateClick = (touchPoint->time - m_start.time) < contextMenuThreshold;
#endif

            if (generateClick) {
                m_gesturedEvent = GesturedEvent::None;

                ClickEvent generatedEvent;
                generatedEvent.event = {
                    wpe_input_pointer_event_type_null, touchPoint->time, touchPoint->x, touchPoint->y,
                    0, 0, 0,
                };
                return generatedEvent;
            }

            FALLTHROUGH;
        }
        case GesturedEvent::ContextMenu:
        {
            m_gesturedEvent = GesturedEvent::None;

            ContextMenuEvent generatedEvent;
            generatedEvent.event = {
                wpe_input_pointer_event_type_null, touchPoint->time, touchPoint->x, touchPoint->y,
                0, 0, 0,
            };
            return generatedEvent;
        }
        case GesturedEvent::Axis:
        {
            m_gesturedEvent = GesturedEvent::None;

            AxisEvent generatedEvent;
            generatedEvent.phase = WebWheelEvent::Phase::PhaseEnded;

#if WPE_CHECK_VERSION(1, 5, 0)
            generatedEvent.event.base = {
                static_cast<enum wpe_input_axis_event_type>(wpe_input_axis_event_type_mask_2d | wpe_input_axis_event_type_motion_smooth),
                touchPoint->time, m_start.x, m_start.y,
                0, 0, 0
            };
            generatedEvent.event.x_axis = generatedEvent.event.y_axis = 0;
#else
            generatedEvent.event = {
                wpe_input_axis_event_type_motion,
                touchPoint->time, m_start.x, m_start.y,
                0, 0, 0
            };
#endif
            m_offset.x = m_offset.y = 0;
            return generatedEvent;
        }
        }
        break;
    }
    default:
        break;
    }

    return NoEvent { };
}

} // namespace WebKit

#endif // USE(LIBWPE) && ENABLE(TOUCH_EVENTS)
