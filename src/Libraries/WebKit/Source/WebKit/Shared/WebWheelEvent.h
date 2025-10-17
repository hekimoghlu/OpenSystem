/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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

#include "WebEvent.h"
#include <WebCore/FloatSize.h>
#include <WebCore/IntPoint.h>

namespace WebKit {

class WebWheelEvent : public WebEvent {
public:
    enum Granularity : uint8_t {
        ScrollByPageWheelEvent,
        ScrollByPixelWheelEvent
    };

    enum Phase : uint32_t {
        PhaseNone        = 0,
        PhaseBegan       = 1 << 0,
        PhaseStationary  = 1 << 1,
        PhaseChanged     = 1 << 2,
        PhaseEnded       = 1 << 3,
        PhaseCancelled   = 1 << 4,
        PhaseMayBegin    = 1 << 5,
    };

    enum class MomentumEndType : uint8_t {
        Unknown,
        Interrupted,
        Natural,
    };

    WebWheelEvent(WebEvent&&, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, const WebCore::FloatSize& delta, const WebCore::FloatSize& wheelTicks, Granularity);
#if PLATFORM(COCOA)
    WebWheelEvent(WebEvent&&, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, const WebCore::FloatSize& delta, const WebCore::FloatSize& wheelTicks, Granularity, bool directionInvertedFromDevice, Phase, Phase momentumPhase, bool hasPreciseScrollingDeltas, uint32_t scrollCount, const WebCore::FloatSize& unacceleratedScrollingDelta, WallTime ioHIDEventTimestamp, std::optional<WebCore::FloatSize> rawPlatformDelta, MomentumEndType);
#elif PLATFORM(GTK) || USE(LIBWPE)
    WebWheelEvent(WebEvent&&, const WebCore::IntPoint& position, const WebCore::IntPoint& globalPosition, const WebCore::FloatSize& delta, const WebCore::FloatSize& wheelTicks, Granularity, Phase, Phase momentumPhase, bool hasPreciseScrollingDeltas);
#endif

    const WebCore::IntPoint position() const { return m_position; }
    void setPosition(WebCore::IntPoint position) { m_position = position; }
    const WebCore::IntPoint globalPosition() const { return m_globalPosition; }
    const WebCore::FloatSize delta() const { return m_delta; }
    const WebCore::FloatSize wheelTicks() const { return m_wheelTicks; }
    Granularity granularity() const { return static_cast<Granularity>(m_granularity); }
    bool directionInvertedFromDevice() const { return m_directionInvertedFromDevice; }
    Phase phase() const { return static_cast<Phase>(m_phase); }
    Phase momentumPhase() const { return static_cast<Phase>(m_momentumPhase); }
    MomentumEndType momentumEndType() const { return m_momentumEndType; }
#if PLATFORM(COCOA) || PLATFORM(GTK) || USE(LIBWPE)
    bool hasPreciseScrollingDeltas() const { return m_hasPreciseScrollingDeltas; }
#endif
#if PLATFORM(COCOA)
    WallTime ioHIDEventTimestamp() const { return m_ioHIDEventTimestamp; }
    std::optional<WebCore::FloatSize> rawPlatformDelta() const { return m_rawPlatformDelta; }
    uint32_t scrollCount() const { return m_scrollCount; }
    const WebCore::FloatSize& unacceleratedScrollingDelta() const { return m_unacceleratedScrollingDelta; }
#endif

private:
    static bool isWheelEventType(WebEventType);

    WebCore::IntPoint m_position;
    WebCore::IntPoint m_globalPosition;
    WebCore::FloatSize m_delta;
    WebCore::FloatSize m_wheelTicks;
    uint32_t m_granularity { ScrollByPageWheelEvent };
    uint32_t m_phase { Phase::PhaseNone };
    uint32_t m_momentumPhase { Phase::PhaseNone };

    MomentumEndType m_momentumEndType { MomentumEndType::Unknown };
    bool m_directionInvertedFromDevice { false };
#if PLATFORM(COCOA) || PLATFORM(GTK) || USE(LIBWPE)
    bool m_hasPreciseScrollingDeltas { false };
#endif
#if PLATFORM(COCOA)
    WallTime m_ioHIDEventTimestamp;
    std::optional<WebCore::FloatSize> m_rawPlatformDelta;
    uint32_t m_scrollCount { 0 };
    WebCore::FloatSize m_unacceleratedScrollingDelta;
#endif
};

} // namespace WebKit
