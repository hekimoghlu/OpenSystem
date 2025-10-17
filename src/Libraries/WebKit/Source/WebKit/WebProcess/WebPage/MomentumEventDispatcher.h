/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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

#if ENABLE(MOMENTUM_EVENT_DISPATCHER)

#define ENABLE_MOMENTUM_EVENT_DISPATCHER_TEMPORARY_LOGGING 0

#include "ScrollingAccelerationCurve.h"
#include "WebWheelEvent.h"
#include <WebCore/FloatSize.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/RectEdges.h>
#include <memory>
#include <wtf/CheckedPtr.h>
#include <wtf/Deque.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
struct DisplayUpdate;
using FramesPerSecond = unsigned;
using PlatformDisplayID = uint32_t;
}

namespace WebKit {

class MomentumEventDispatcher {
    WTF_MAKE_NONCOPYABLE(MomentumEventDispatcher);
    WTF_MAKE_TZONE_ALLOCATED(MomentumEventDispatcher);
public:
    class Client : public CanMakeCheckedPtr<Client> {
        WTF_MAKE_FAST_ALLOCATED;
        WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(Client);
    friend class MomentumEventDispatcher;
    public:
        virtual ~Client() = default;

    private:
        virtual void handleSyntheticWheelEvent(WebCore::PageIdentifier, const WebWheelEvent&, WebCore::RectEdges<bool> rubberBandableEdges) = 0;
        
        virtual void startDisplayDidRefreshCallbacks(WebCore::PlatformDisplayID) = 0;
        virtual void stopDisplayDidRefreshCallbacks(WebCore::PlatformDisplayID) = 0;

#if ENABLE(MOMENTUM_EVENT_DISPATCHER_TEMPORARY_LOGGING)
        virtual void flushMomentumEventLoggingSoon() = 0;
#endif
    };

    MomentumEventDispatcher(Client&);
    ~MomentumEventDispatcher();

    bool handleWheelEvent(WebCore::PageIdentifier, const WebWheelEvent&, WebCore::RectEdges<bool> rubberBandableEdges);

    void setScrollingAccelerationCurve(WebCore::PageIdentifier, std::optional<ScrollingAccelerationCurve>);

    void displayDidRefresh(WebCore::PlatformDisplayID);

    void pageScreenDidChange(WebCore::PageIdentifier, WebCore::PlatformDisplayID, std::optional<unsigned> nominalFramesPerSecond);

#if ENABLE(MOMENTUM_EVENT_DISPATCHER_TEMPORARY_LOGGING)
    void flushLog();
#endif

private:
    void didStartMomentumPhase(WebCore::PageIdentifier, const WebWheelEvent&);
    void didEndMomentumPhase();

    bool eventShouldStartSyntheticMomentumPhase(WebCore::PageIdentifier, const WebWheelEvent&) const;

    std::optional<ScrollingAccelerationCurve> scrollingAccelerationCurveForPage(WebCore::PageIdentifier) const;

    void startDisplayLink();
    void stopDisplayLink();

    struct DisplayProperties {
        WebCore::PlatformDisplayID displayID;
        WebCore::FramesPerSecond nominalFrameRate;
    };
    std::optional<DisplayProperties> displayProperties(WebCore::PageIdentifier) const;

    void dispatchSyntheticMomentumEvent(WebWheelEvent::Phase, WebCore::FloatSize delta);

    void buildOffsetTableWithInitialDelta(WebCore::FloatSize);
    void equalizeTailGaps();

    // Once consumed, this delta *must* be dispatched in an event.
    std::optional<WebCore::FloatSize> consumeDeltaForCurrentTime();

    WebCore::FloatSize offsetAtTime(Seconds);
    std::pair<WebCore::FloatSize, WebCore::FloatSize> computeNextDelta(WebCore::FloatSize currentUnacceleratedDelta);

    void didReceiveScrollEventWithInterval(WebCore::FloatSize, Seconds);
    void didReceiveScrollEvent(const WebWheelEvent&);

#if ENABLE(MOMENTUM_EVENT_DISPATCHER_TEMPORARY_LOGGING)
    void pushLogEntry(uint32_t generatedPhase, uint32_t eventPhase);

    WebCore::FloatSize m_lastActivePhaseDelta;

    struct LogEntry {
        MonotonicTime time;

        float totalGeneratedOffset { 0 };
        float totalEventOffset { 0 };

        uint32_t generatedPhase { 0 };
        uint32_t eventPhase { 0 };
    };
    LogEntry m_currentLogState;
    Vector<LogEntry> m_log;
#endif

    struct Delta {
        float rawPlatformDelta;
        Seconds frameInterval;
    };
    static constexpr unsigned deltaHistoryQueueSize = 9;
    typedef Deque<Delta, deltaHistoryQueueSize> HistoricalDeltas;
    HistoricalDeltas m_deltaHistoryX;
    HistoricalDeltas m_deltaHistoryY;

    Markable<WallTime> m_lastScrollTimestamp;
    std::optional<WebWheelEvent> m_lastIncomingEvent;
    WebCore::RectEdges<bool> m_lastRubberBandableEdges;
    bool m_isInOverriddenPlatformMomentumGesture { false };

    struct {
        bool active { false };

        Markable<WebCore::PageIdentifier> pageIdentifier;
        std::optional<ScrollingAccelerationCurve> accelerationCurve;
        std::optional<WebWheelEvent> initiatingEvent;

        WebCore::FloatSize currentOffset;
        MonotonicTime startTime;

        Vector<WebCore::FloatSize> offsetTable; // Always at 60Hz intervals.
        Vector<WebCore::FloatSize> tailDeltaTable; // Always at event dispatch intervals.
        Seconds tailStartDelay;
        unsigned currentTailDeltaIndex { 0 };

        WebCore::FramesPerSecond displayNominalFrameRate { 0 };

#if ENABLE(MOMENTUM_EVENT_DISPATCHER_TEMPORARY_LOGGING)
        WebCore::FloatSize accumulatedEventOffset;
        bool didLogInitialQueueState { false };
#endif

#if ENABLE(MOMENTUM_EVENT_DISPATCHER_PREMATURE_ROUNDING)
        WebCore::FloatSize carryOffset;
#endif
    } m_currentGesture;

    HashMap<WebCore::PageIdentifier, DisplayProperties> m_displayProperties;

    mutable Lock m_accelerationCurvesLock;
    HashMap<WebCore::PageIdentifier, std::optional<ScrollingAccelerationCurve>> m_accelerationCurves WTF_GUARDED_BY_LOCK(m_accelerationCurvesLock);
    CheckedRef<Client> m_client;
};

} // namespace WebKit

#endif
