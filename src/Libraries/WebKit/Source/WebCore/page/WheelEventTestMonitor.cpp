/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#include "WheelEventTestMonitor.h"

#include "Logging.h"
#include "Page.h"
#include <wtf/OptionSet.h>
#include <wtf/RunLoop.h>
#include <wtf/text/TextStream.h>

#if !LOG_DISABLED
#include <wtf/text/CString.h>
#include <wtf/text/StringBuilder.h>
#endif

namespace WebCore {

WheelEventTestMonitor::WheelEventTestMonitor(Page& page)
    : m_page(page)
{
}

void WheelEventTestMonitor::clearAllTestDeferrals()
{
    Locker locker { m_lock };

    ASSERT(isMainThread());
    m_deferCompletionReasons.clear();
    m_completionCallback = nullptr;
    m_everHadDeferral = false;
    m_receivedWheelEndOrCancel = false;
    m_receivedMomentumEnd = false;
    LOG_WITH_STREAM(WheelEventTestMonitor, stream << "  WheelEventTestMonitor::clearAllTestDeferrals: cleared all test state.");
}

void WheelEventTestMonitor::setTestCallbackAndStartMonitoring(bool expectWheelEndOrCancel, bool expectMomentumEnd, Function<void()>&& functionCallback)
{
    Locker locker { m_lock };

    ASSERT(isMainThread());
    m_completionCallback = WTFMove(functionCallback);
#if ENABLE(KINETIC_SCROLLING)
    m_expectWheelEndOrCancel = expectWheelEndOrCancel;
    m_expectMomentumEnd = expectMomentumEnd;
#else
    UNUSED_PARAM(expectWheelEndOrCancel);
    UNUSED_PARAM(expectMomentumEnd);
#endif

    m_page.scheduleRenderingUpdate(RenderingUpdateStep::WheelEventMonitorCallbacks);

    LOG_WITH_STREAM(WheelEventTestMonitor, stream << "  WheelEventTestMonitor::setTestCallbackAndStartMonitoring - expect end/cancel " << expectWheelEndOrCancel << ", expect momentum end " << expectMomentumEnd);
}

void WheelEventTestMonitor::deferForReason(ScrollingNodeID identifier, OptionSet<DeferReason> reason)
{
    Locker locker { m_lock };

    m_deferCompletionReasons.ensure(identifier, [] {
        return OptionSet<DeferReason>();
    }).iterator->value.add(reason);

    m_everHadDeferral = true;

    LOG_WITH_STREAM(WheelEventTestMonitor, stream << "      (=) WheelEventTestMonitor::deferForReason: id=" << identifier << ", reason=" << reason);
}

void WheelEventTestMonitor::removeDeferralForReason(ScrollingNodeID identifier, OptionSet<DeferReason> reason)
{
    Locker locker { m_lock };

    auto it = m_deferCompletionReasons.find(identifier);
    if (it == m_deferCompletionReasons.end()) {
        LOG_WITH_STREAM(WheelEventTestMonitor, stream << "      (=) WheelEventTestMonitor::removeDeferralForReason: failed to find defer for id=" << identifier << ", reason=" << reason);
        return;
    }

    LOG_WITH_STREAM(WheelEventTestMonitor, stream << "      (=) WheelEventTestMonitor::removeDeferralForReason: id=" << identifier << ", reason=" << reason);
    it->value.remove(reason);
    
    if (it->value.isEmpty())
        m_deferCompletionReasons.remove(it);

    scheduleCallbackCheck();
}

void WheelEventTestMonitor::receivedWheelEventWithPhases(PlatformWheelEventPhase phase, PlatformWheelEventPhase momentumPhase)
{
#if ENABLE(KINETIC_SCROLLING)
    Locker locker { m_lock };

    LOG_WITH_STREAM(WheelEventTestMonitor, stream << "      (=) WheelEventTestMonitor::receivedWheelEventWithPhases: phase=" << phase << " momentumPhase=" << momentumPhase);

    if (phase == PlatformWheelEventPhase::Ended || phase == PlatformWheelEventPhase::Cancelled)
        m_receivedWheelEndOrCancel = true;

    if (momentumPhase == PlatformWheelEventPhase::Ended)
        m_receivedMomentumEnd = true;
#else
    UNUSED_PARAM(phase);
    UNUSED_PARAM(momentumPhase);
#endif
}

void WheelEventTestMonitor::scheduleCallbackCheck()
{
    ensureOnMainThread([weakThis = ThreadSafeWeakPtr { *this }] {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return;
        protectedThis->m_page.scheduleRenderingUpdate(RenderingUpdateStep::WheelEventMonitorCallbacks);
    });
}

void WheelEventTestMonitor::checkShouldFireCallbacks()
{
    ASSERT(isMainThread());
    {
        Locker locker { m_lock };

        if (!m_deferCompletionReasons.isEmpty()) {
            LOG_WITH_STREAM(WheelEventTestMonitor, stream << "  WheelEventTestMonitor::checkShouldFireCallbacks - scrolling still active, reasons " << m_deferCompletionReasons);
            return;
        }

        if (!m_everHadDeferral) {
            LOG_WITH_STREAM(WheelEventTestMonitor, stream << "  WheelEventTestMonitor::checkShouldFireCallbacks - have not yet seen any deferral reasons");
            return;
        }
        
        if (m_expectWheelEndOrCancel && !m_receivedWheelEndOrCancel) {
            LOG_WITH_STREAM(WheelEventTestMonitor, stream << "  WheelEventTestMonitor::checkShouldFireCallbacks - have not seen end of of wheel phase");
            return;
        }

        if (m_expectMomentumEnd && !m_receivedMomentumEnd) {
            LOG_WITH_STREAM(WheelEventTestMonitor, stream << "  WheelEventTestMonitor::checkShouldFireCallbacks - have not seen end of of momentum phase");
            return;
        }
    }

    if (auto functionCallback = WTFMove(m_completionCallback)) {
        LOG_WITH_STREAM(WheelEventTestMonitor, stream << "  WheelEventTestMonitor::checkShouldFireCallbacks: scrolling is idle, FIRING TEST");
        functionCallback();
    } else
        LOG_WITH_STREAM(WheelEventTestMonitor, stream << "  WheelEventTestMonitor::checkShouldFireCallbacks - no callback");
}

TextStream& operator<<(TextStream& ts, WheelEventTestMonitor::DeferReason reason)
{
    switch (reason) {
    case WheelEventTestMonitor::DeferReason::HandlingWheelEvent: ts << "handling wheel event"; break;
    case WheelEventTestMonitor::DeferReason::HandlingWheelEventOnMainThread: ts << "handling wheel event on main thread"; break;
    case WheelEventTestMonitor::DeferReason::PostMainThreadWheelEventHandling: ts << "post-main thread event handling"; break;
    case WheelEventTestMonitor::DeferReason::RubberbandInProgress: ts << "rubberbanding"; break;
    case WheelEventTestMonitor::DeferReason::ScrollSnapInProgress: ts << "scroll-snapping"; break;
    case WheelEventTestMonitor::DeferReason::ScrollAnimationInProgress: ts << "scroll animation"; break;
    case WheelEventTestMonitor::DeferReason::ScrollingThreadSyncNeeded: ts << "scrolling thread sync needed"; break;
    case WheelEventTestMonitor::DeferReason::ContentScrollInProgress: ts << "content scrolling"; break;
    case WheelEventTestMonitor::DeferReason::RequestedScrollPosition: ts << "requested scroll position"; break;
    case WheelEventTestMonitor::DeferReason::CommittingTransientZoom: ts << "committing transient zoom"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, const WheelEventTestMonitor::ScrollableAreaReasonMap& reasonMap)
{
    for (const auto& regionReasonsPair : reasonMap)
        ts << "   scroll region: " << regionReasonsPair.key << " reasons: " << regionReasonsPair.value;

    return ts;
}

} // namespace WebCore
