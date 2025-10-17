/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#import "config.h"

#if PLATFORM(MAC)
#import "WheelEventDeltaFilterMac.h"

#import "FloatPoint.h"
#import "Logging.h"
#import "PlatformWheelEvent.h"
#import <pal/spi/mac/NSScrollingInputFilterSPI.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WheelEventDeltaFilterMac);

WheelEventDeltaFilterMac::WheelEventDeltaFilterMac()
    : WheelEventDeltaFilter()
    , m_predominantAxisFilter(adoptNS([[_NSScrollingPredominantAxisFilter alloc] init]))
    , m_initialWallTime(WallTime::now())
{
}

void WheelEventDeltaFilterMac::updateFromEvent(const PlatformWheelEvent& event)
{
    if (event.momentumPhase() != PlatformWheelEventPhase::None) {
        if (event.momentumPhase() == PlatformWheelEventPhase::Began)
            updateCurrentVelocityFromEvent(event);
        m_lastIOHIDEventTimestamp = event.ioHIDEventTimestamp();
        return;
    }

    switch (event.phase()) {
    case PlatformWheelEventPhase::None:
    case PlatformWheelEventPhase::Ended:
        break;

    case PlatformWheelEventPhase::Began:
        reset();
        updateCurrentVelocityFromEvent(event);
        break;

    case PlatformWheelEventPhase::Changed:
        updateCurrentVelocityFromEvent(event);
        break;

    case PlatformWheelEventPhase::MayBegin:
    case PlatformWheelEventPhase::Cancelled:
    case PlatformWheelEventPhase::Stationary:
        reset();
        break;
    }

    m_lastIOHIDEventTimestamp = event.ioHIDEventTimestamp();
}

void WheelEventDeltaFilterMac::updateCurrentVelocityFromEvent(const PlatformWheelEvent& event)
{
    // The absolute value of timestamp doesn't matter; the filter looks at deltas from the previous event.
    auto timestamp = event.timestamp() - m_initialWallTime;

    NSPoint filteredDeltaResult;
    NSPoint filteredVelocityResult;

    [m_predominantAxisFilter filterInputDelta:toFloatPoint(event.delta()) timestamp:timestamp.seconds() outputDelta:&filteredDeltaResult velocity:&filteredVelocityResult];
    auto axisFilteredVelocity = toFloatSize(filteredVelocityResult);
    m_currentFilteredDelta = toFloatSize(filteredDeltaResult);

    // Use a 1ms minimum to avoid divide by zero. The usual cadence of these events matches screen refresh rate.
    auto deltaFromLastEvent = std::max(event.ioHIDEventTimestamp() - m_lastIOHIDEventTimestamp, 1_ms);
    m_currentFilteredVelocity = event.delta() / deltaFromLastEvent.seconds();

    // Apply the axis-locking that m_predominantAxisFilter does.
    if (!axisFilteredVelocity.width())
        m_currentFilteredVelocity.setWidth(0);
    if (!axisFilteredVelocity.height())
        m_currentFilteredVelocity.setHeight(0);

    LOG(ScrollAnimations, "WheelEventDeltaFilterMac::updateFromEvent: _NSScrollingPredominantAxisFilter velocity %.2f, %2f, IOHIDEvent velocity %.2f,%.2f",
        axisFilteredVelocity.width(), axisFilteredVelocity.height(), m_currentFilteredVelocity.width(), m_currentFilteredVelocity.height());
}

void WheelEventDeltaFilterMac::reset()
{
    [m_predominantAxisFilter reset];
    m_currentFilteredVelocity = { };
    m_currentFilteredDelta = { };
    m_lastIOHIDEventTimestamp = { };
}

}

#endif /* PLATFORM(MAC) */
