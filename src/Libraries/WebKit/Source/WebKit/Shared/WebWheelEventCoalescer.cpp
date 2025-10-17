/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#include "WebWheelEventCoalescer.h"

#include "Logging.h"
#include "NativeWebWheelEvent.h"
#include "WebEventConversion.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebKit {

// Represents the number of wheel events we can hold in the queue before we start pushing them preemptively.
constexpr unsigned wheelEventQueueSizeThreshold = 10;

#if !LOG_DISABLED
static TextStream& operator<<(TextStream& ts, const NativeWebWheelEvent& nativeWheelEvent)
{
    ts << platform(nativeWheelEvent);
    return ts;
}
#endif

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebWheelEventCoalescer);

bool WebWheelEventCoalescer::canCoalesce(const WebWheelEvent& a, const WebWheelEvent& b)
{
    if (a.position() != b.position())
        return false;
    if (a.globalPosition() != b.globalPosition())
        return false;
    if (a.modifiers() != b.modifiers())
        return false;
    if (a.granularity() != b.granularity())
        return false;
#if PLATFORM(COCOA)
    if (a.phase() != b.phase())
        return false;
    if (a.momentumPhase() != b.momentumPhase())
        return false;
#endif
#if PLATFORM(COCOA) || PLATFORM(GTK) || USE(LIBWPE)
    if (a.hasPreciseScrollingDeltas() != b.hasPreciseScrollingDeltas())
        return false;
#endif

    return true;
}

WebWheelEvent WebWheelEventCoalescer::coalesce(const WebWheelEvent& a, const WebWheelEvent& b)
{
    ASSERT(canCoalesce(a, b));

    auto mergedDelta = a.delta() + b.delta();
    auto mergedWheelTicks = a.wheelTicks() + b.wheelTicks();

#if PLATFORM(COCOA)
    auto mergedUnacceleratedScrollingDelta = a.unacceleratedScrollingDelta() + b.unacceleratedScrollingDelta();
    std::optional<WebCore::FloatSize> mergedRawPlatformScrollingDelta;
    if (a.rawPlatformDelta() && b.rawPlatformDelta())
        mergedRawPlatformScrollingDelta = a.rawPlatformDelta().value() + b.rawPlatformDelta().value();

    auto event = WebWheelEvent({ WebEventType::Wheel, b.modifiers(), b.timestamp() }, b.position(), b.globalPosition(), mergedDelta, mergedWheelTicks, b.granularity(), b.directionInvertedFromDevice(), b.phase(), b.momentumPhase(), b.hasPreciseScrollingDeltas(), b.scrollCount(), mergedUnacceleratedScrollingDelta, b.ioHIDEventTimestamp(), mergedRawPlatformScrollingDelta, b.momentumEndType());
#elif PLATFORM(GTK) || USE(LIBWPE)
    auto event = WebWheelEvent({ WebEventType::Wheel, b.modifiers(), b.timestamp() }, b.position(), b.globalPosition(), mergedDelta, mergedWheelTicks, b.granularity(), b.phase(), b.momentumPhase(), b.hasPreciseScrollingDeltas());
#else
    auto event = WebWheelEvent({ WebEventType::Wheel, b.modifiers(), b.timestamp() }, b.position(), b.globalPosition(), mergedDelta, mergedWheelTicks, b.granularity());
#endif
    return event;
}

bool WebWheelEventCoalescer::shouldDispatchEventNow(const WebWheelEvent& event) const
{
#if PLATFORM(GTK)
    // Don't queue events representing a non-trivial scrolling phase to
    // avoid having them trapped in the queue, potentially preventing a
    // scrolling session to beginning or end correctly.
    // This is only needed by platforms whose WebWheelEvent has this phase
    // information (Cocoa and GTK+) but Cocoa was fine without it.
    if (event.phase() == WebWheelEvent::Phase::PhaseNone
        || event.phase() == WebWheelEvent::Phase::PhaseChanged
        || event.momentumPhase() == WebWheelEvent::Phase::PhaseNone
        || event.momentumPhase() == WebWheelEvent::Phase::PhaseChanged)
        return true;
#else
    UNUSED_PARAM(event);
#endif

    return m_wheelEventQueue.size() >= wheelEventQueueSizeThreshold;
}

std::optional<WebWheelEvent> WebWheelEventCoalescer::nextEventToDispatch()
{
    if (m_wheelEventQueue.isEmpty())
        return std::nullopt;

    auto coalescedNativeEvent = m_wheelEventQueue.takeFirst();

    auto coalescedSequence = makeUnique<CoalescedEventSequence>();
    coalescedSequence->append(coalescedNativeEvent);

    auto coalescedWebEvent = WebWheelEvent { coalescedNativeEvent };

    while (!m_wheelEventQueue.isEmpty() && canCoalesce(coalescedWebEvent, m_wheelEventQueue.first())) {
        auto firstEvent = m_wheelEventQueue.takeFirst();
        coalescedSequence->append(firstEvent);
        coalescedWebEvent = coalesce(coalescedWebEvent, WebWheelEvent { firstEvent });
    }

#if !LOG_DISABLED
    if (coalescedSequence->size() > 1)
        LOG_WITH_STREAM(WheelEvents, stream << "WebWheelEventCoalescer::wheelEventWithCoalescing coalesced " << *coalescedSequence << " into " << platform(coalescedWebEvent));
#endif

    m_eventsBeingProcessed.append(WTFMove(coalescedSequence));
    return coalescedWebEvent;
}

bool WebWheelEventCoalescer::shouldDispatchEvent(const NativeWebWheelEvent& event)
{
    LOG_WITH_STREAM(WheelEvents, stream << "WebWheelEventCoalescer::shouldDispatchEvent " << platform(event) << " (" << m_wheelEventQueue.size() << " events in the queue, " << m_eventsBeingProcessed.size() << " event sequences being processed)");

    m_wheelEventQueue.append(event);

    if (!m_eventsBeingProcessed.isEmpty()) {
        if (!shouldDispatchEventNow(m_wheelEventQueue.last())) {
            LOG_WITH_STREAM(WheelEvents, stream << "WebWheelEventCoalescer::shouldDispatchEvent -  " << m_wheelEventQueue.size() << " events queued; not dispatching");
            return false;
        }
        // The queue has too many wheel events, so push a new event.
        // FIXME: This logic is confusing, and possibly not necessary.
    }

    return true;
}

std::optional<NativeWebWheelEvent> WebWheelEventCoalescer::takeOldestEventBeingProcessed()
{
    if (m_eventsBeingProcessed.isEmpty())
        return { };

    auto oldestSequence = m_eventsBeingProcessed.takeFirst();
    return oldestSequence->last();
}

void WebWheelEventCoalescer::clear()
{
    m_wheelEventQueue.clear();
    m_eventsBeingProcessed.clear();
}

} // namespace WebKit
