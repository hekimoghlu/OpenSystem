/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#include "WebWheelEvent.h"


namespace WebKit {

using namespace WebCore;

WebWheelEvent::WebWheelEvent(WebEvent&& event, const IntPoint& position, const IntPoint& globalPosition, const FloatSize& delta, const FloatSize& wheelTicks, Granularity granularity)
    : WebEvent(WTFMove(event))
    , m_position(position)
    , m_globalPosition(globalPosition)
    , m_delta(delta)
    , m_wheelTicks(wheelTicks)
    , m_granularity(granularity)
{
    ASSERT(isWheelEventType(type()));
}

#if PLATFORM(COCOA)
WebWheelEvent::WebWheelEvent(WebEvent&& event, const IntPoint& position, const IntPoint& globalPosition, const FloatSize& delta, const FloatSize& wheelTicks, Granularity granularity, bool directionInvertedFromDevice, Phase phase, Phase momentumPhase, bool hasPreciseScrollingDeltas, uint32_t scrollCount, const WebCore::FloatSize& unacceleratedScrollingDelta, WallTime ioHIDEventTimestamp, std::optional<WebCore::FloatSize> rawPlatformDelta, MomentumEndType momentumEndType)
    : WebEvent(WTFMove(event))
    , m_position(position)
    , m_globalPosition(globalPosition)
    , m_delta(delta)
    , m_wheelTicks(wheelTicks)
    , m_granularity(granularity)
    , m_phase(phase)
    , m_momentumPhase(momentumPhase)
    , m_momentumEndType(momentumEndType)
    , m_directionInvertedFromDevice(directionInvertedFromDevice)
    , m_hasPreciseScrollingDeltas(hasPreciseScrollingDeltas)
    , m_ioHIDEventTimestamp(ioHIDEventTimestamp)
    , m_rawPlatformDelta(rawPlatformDelta)
    , m_scrollCount(scrollCount)
    , m_unacceleratedScrollingDelta(unacceleratedScrollingDelta)
{
    ASSERT(isWheelEventType(type()));
}
#elif PLATFORM(GTK) || USE(LIBWPE)
WebWheelEvent::WebWheelEvent(WebEvent&& event, const IntPoint& position, const IntPoint& globalPosition, const FloatSize& delta, const FloatSize& wheelTicks, Granularity granularity, Phase phase, Phase momentumPhase, bool hasPreciseScrollingDeltas)
    : WebEvent(WTFMove(event))
    , m_position(position)
    , m_globalPosition(globalPosition)
    , m_delta(delta)
    , m_wheelTicks(wheelTicks)
    , m_granularity(granularity)
    , m_phase(phase)
    , m_momentumPhase(momentumPhase)
    , m_hasPreciseScrollingDeltas(hasPreciseScrollingDeltas)
{
    ASSERT(isWheelEventType(type()));
}
#endif

bool WebWheelEvent::isWheelEventType(WebEventType type)
{
    return type == WebEventType::Wheel;
}

} // namespace WebKit
