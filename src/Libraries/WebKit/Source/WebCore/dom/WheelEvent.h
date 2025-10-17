/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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

#include "MouseEvent.h"
#include "PlatformWheelEvent.h"

namespace WebCore {

class WheelEvent final : public MouseEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WheelEvent);
public:
    static constexpr int TickMultiplier = 120;

    enum {
        DOM_DELTA_PIXEL = 0,
        DOM_DELTA_LINE,
        DOM_DELTA_PAGE
    };

    static Ref<WheelEvent> create(const PlatformWheelEvent&, RefPtr<WindowProxy>&&, IsCancelable = IsCancelable::Yes);
    static Ref<WheelEvent> createForBindings();

    struct Init : MouseEventInit {
        double deltaX { 0 };
        double deltaY { 0 };
        double deltaZ { 0 };
        unsigned deltaMode { DOM_DELTA_PIXEL };
        int wheelDeltaX { 0 }; // Deprecated.
        int wheelDeltaY { 0 }; // Deprecated.
    };

    static Ref<WheelEvent> create(const AtomString& type, const Init&);

    const std::optional<PlatformWheelEvent>& underlyingPlatformEvent() const { return m_underlyingPlatformEvent; }

    double deltaX() const { return m_deltaX; } // Positive when scrolling right.
    double deltaY() const { return m_deltaY; } // Positive when scrolling down.
    double deltaZ() const { return m_deltaZ; }
    int wheelDelta() const { return wheelDeltaY() ? wheelDeltaY() : wheelDeltaX(); } // Deprecated.
    int wheelDeltaX() const { return m_wheelDelta.x(); } // Deprecated, negative when scrolling right.
    int wheelDeltaY() const { return m_wheelDelta.y(); } // Deprecated, negative when scrolling down.
    unsigned deltaMode() const { return m_deltaMode; }

    bool webkitDirectionInvertedFromDevice() const { return m_underlyingPlatformEvent && m_underlyingPlatformEvent.value().directionInvertedFromDevice(); }

#if PLATFORM(MAC)
    PlatformWheelEventPhase phase() const { return m_underlyingPlatformEvent ? m_underlyingPlatformEvent.value().phase() : PlatformWheelEventPhase::None; }
    PlatformWheelEventPhase momentumPhase() const { return m_underlyingPlatformEvent ? m_underlyingPlatformEvent.value().momentumPhase() : PlatformWheelEventPhase::None; }
#endif

private:
    WheelEvent();
    WheelEvent(const AtomString&, const Init&);
    WheelEvent(const PlatformWheelEvent&, RefPtr<WindowProxy>&&, IsCancelable);

    bool isWheelEvent() const final;

    IntPoint m_wheelDelta;
    double m_deltaX { 0 };
    double m_deltaY { 0 };
    double m_deltaZ { 0 };
    unsigned m_deltaMode { DOM_DELTA_PIXEL };
    std::optional<PlatformWheelEvent> m_underlyingPlatformEvent;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(WheelEvent)
