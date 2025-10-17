/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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

#include "PlatformWheelEvent.h"
#include "ScrollingNodeID.h"
#include <functional>
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

class Page;

enum class WheelEventTestMonitorDeferReason : uint16_t {
    HandlingWheelEvent                  = 1 << 0,
    HandlingWheelEventOnMainThread      = 1 << 1,
    PostMainThreadWheelEventHandling    = 1 << 2,
    RubberbandInProgress                = 1 << 3,
    ScrollSnapInProgress                = 1 << 4,
    ScrollAnimationInProgress           = 1 << 5,
    ScrollingThreadSyncNeeded           = 1 << 6,
    ContentScrollInProgress             = 1 << 7,
    RequestedScrollPosition             = 1 << 8,
    CommittingTransientZoom             = 1 << 9,
};

class WheelEventTestMonitor : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<WheelEventTestMonitor> {
public:
    WheelEventTestMonitor(Page&);

    WEBCORE_EXPORT void setTestCallbackAndStartMonitoring(bool expectWheelEndOrCancel, bool expectMomentumEnd, Function<void()>&&);
    WEBCORE_EXPORT void clearAllTestDeferrals();
    
    using DeferReason = WheelEventTestMonitorDeferReason;

    WEBCORE_EXPORT void receivedWheelEventWithPhases(PlatformWheelEventPhase phase, PlatformWheelEventPhase momentumPhase);
    WEBCORE_EXPORT void deferForReason(ScrollingNodeID, OptionSet<DeferReason>);
    WEBCORE_EXPORT void removeDeferralForReason(ScrollingNodeID, OptionSet<DeferReason>);
    
    void checkShouldFireCallbacks();

    using ScrollableAreaReasonMap = UncheckedKeyHashMap<ScrollingNodeID, OptionSet<DeferReason>>;

private:
    void scheduleCallbackCheck();

    Function<void()> m_completionCallback;
    Page& m_page;

    Lock m_lock;
    ScrollableAreaReasonMap m_deferCompletionReasons WTF_GUARDED_BY_LOCK(m_lock);
    bool m_expectWheelEndOrCancel WTF_GUARDED_BY_LOCK(m_lock) { false };
    bool m_receivedWheelEndOrCancel WTF_GUARDED_BY_LOCK(m_lock) { false };
    bool m_expectMomentumEnd WTF_GUARDED_BY_LOCK(m_lock) { false };
    bool m_receivedMomentumEnd WTF_GUARDED_BY_LOCK(m_lock) { false };
    bool m_everHadDeferral WTF_GUARDED_BY_LOCK(m_lock) { false };
};

class WheelEventTestMonitorCompletionDeferrer {
    WTF_MAKE_NONCOPYABLE(WheelEventTestMonitorCompletionDeferrer);
    WTF_MAKE_FAST_ALLOCATED;
public:
    WheelEventTestMonitorCompletionDeferrer(WheelEventTestMonitor* monitor, ScrollingNodeID identifier, WheelEventTestMonitor::DeferReason reason)
        : m_monitor(monitor)
        , m_identifier(identifier)
        , m_reason(reason)
    {
        if (m_monitor)
            m_monitor->deferForReason(m_identifier, m_reason);
    }
    
    WheelEventTestMonitorCompletionDeferrer(WheelEventTestMonitorCompletionDeferrer&& other)
        : m_monitor(WTFMove(other.m_monitor))
        , m_identifier(other.m_identifier)
        , m_reason(other.m_reason)
    {
    }

    ~WheelEventTestMonitorCompletionDeferrer()
    {
        if (m_monitor)
            m_monitor->removeDeferralForReason(m_identifier, m_reason);
    }

private:
    RefPtr<WheelEventTestMonitor> m_monitor;
    ScrollingNodeID m_identifier;
    WheelEventTestMonitor::DeferReason m_reason;
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, WheelEventTestMonitor::DeferReason);
WTF::TextStream& operator<<(WTF::TextStream&, const WheelEventTestMonitor::ScrollableAreaReasonMap&);

} // namespace WebCore
