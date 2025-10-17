/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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

#if USE(LIBWPE) && ENABLE(TOUCH_EVENTS)

#include "WebWheelEvent.h"
#include <variant>
#include <wpe/wpe.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class TouchGestureController {
    WTF_MAKE_TZONE_ALLOCATED(TouchGestureController);
public:
    TouchGestureController() = default;

    enum class GesturedEvent {
        None,
        Click,
        ContextMenu,
        Axis,
    };

    struct NoEvent { };

    struct ClickEvent {
        struct wpe_input_pointer_event event;
    };

    struct ContextMenuEvent {
        struct wpe_input_pointer_event event;
    };

    struct AxisEvent {
#if WPE_CHECK_VERSION(1, 5, 0)
        struct wpe_input_axis_2d_event event;
#else
        struct wpe_input_axis_event event;
#endif
        WebWheelEvent::Phase phase;
    };

    using EventVariant = std::variant<NoEvent, ClickEvent, ContextMenuEvent, AxisEvent>;

    GesturedEvent gesturedEvent() const { return m_gesturedEvent; }
    EventVariant handleEvent(const struct wpe_input_touch_event_raw*);

private:
    GesturedEvent m_gesturedEvent { GesturedEvent::None };

    struct {
        bool active { false };
        uint32_t time { 0 };
        int32_t x { 0 };
        int32_t y { 0 };
    } m_start;

    struct {
        int32_t x { 0 };
        int32_t y { 0 };
    } m_offset;

    bool m_xAxisLockBroken { false };
    bool m_yAxisLockBroken { false };
};

} // namespace WebKit

#endif // USE(LIBWPE) && ENABLE(TOUCH_EVENTS)
