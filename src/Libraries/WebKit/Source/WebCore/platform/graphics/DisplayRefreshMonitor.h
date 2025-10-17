/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

#include "AnimationFrameRate.h"
#include "DisplayUpdate.h"
#include "PlatformScreen.h"
#include <wtf/CheckedPtr.h>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class DisplayAnimationClient;
class DisplayRefreshMonitorClient;
class DisplayRefreshMonitorFactory;

class DisplayRefreshMonitor : public ThreadSafeRefCounted<DisplayRefreshMonitor> {
    friend class DisplayRefreshMonitorManager;
public:
    static RefPtr<DisplayRefreshMonitor> create(DisplayRefreshMonitorFactory*, PlatformDisplayID);
    WEBCORE_EXPORT virtual ~DisplayRefreshMonitor();
    
    WEBCORE_EXPORT virtual void stop();

    // Return true if callback request was scheduled, false if it couldn't be
    // (e.g., hardware refresh is not available)
    WEBCORE_EXPORT virtual bool requestRefreshCallback();

    void windowScreenDidChange(PlatformDisplayID);
    
    bool hasClients() const { return m_clients.size(); }
    void addClient(DisplayRefreshMonitorClient&);
    bool removeClient(DisplayRefreshMonitorClient&);

    void clientPreferredFramesPerSecondChanged(DisplayRefreshMonitorClient&);
    std::optional<FramesPerSecond> maxClientPreferredFramesPerSecond() const { return m_maxClientPreferredFramesPerSecond; }

    virtual std::optional<FramesPerSecond> displayNominalFramesPerSecond() { return std::nullopt; }

    PlatformDisplayID displayID() const { return m_displayID; }

    static RefPtr<DisplayRefreshMonitor> createDefaultDisplayRefreshMonitor(PlatformDisplayID);
    WEBCORE_EXPORT virtual void displayLinkFired(const DisplayUpdate&);

protected:
    WEBCORE_EXPORT explicit DisplayRefreshMonitor(PlatformDisplayID);

    WEBCORE_EXPORT virtual void dispatchDisplayDidRefresh(const DisplayUpdate&);

    Lock& lock() WTF_RETURNS_LOCK(m_lock) { return m_lock; }
    void setMaxUnscheduledFireCount(unsigned count) WTF_REQUIRES_LOCK(m_lock) { m_maxUnscheduledFireCount = count; }

    // Returns true if the start was successful.
    WEBCORE_EXPORT virtual bool startNotificationMechanism() = 0;
    WEBCORE_EXPORT virtual void stopNotificationMechanism() = 0;

    bool isScheduled() const WTF_REQUIRES_LOCK(m_lock) { return m_scheduled; }
    void setIsScheduled(bool scheduled) WTF_REQUIRES_LOCK(m_lock) { m_scheduled = scheduled; }

    bool isPreviousFrameDone() const WTF_REQUIRES_LOCK(m_lock) { return m_previousFrameDone; }
    void setIsPreviousFrameDone(bool done) WTF_REQUIRES_LOCK(m_lock) { m_previousFrameDone = done; }

    WEBCORE_EXPORT void displayDidRefresh(const DisplayUpdate&);

private:
    bool firedAndReachedMaxUnscheduledFireCount() WTF_REQUIRES_LOCK(m_lock);

    virtual void adjustPreferredFramesPerSecond(FramesPerSecond) { }

    std::optional<FramesPerSecond> maximumClientPreferredFramesPerSecond() const;
    void computeMaxPreferredFramesPerSecond();

    UncheckedKeyHashSet<CheckedPtr<DisplayRefreshMonitorClient>> m_clients;
    UncheckedKeyHashSet<CheckedPtr<DisplayRefreshMonitorClient>>* m_clientsToBeNotified { nullptr };

    PlatformDisplayID m_displayID { 0 };
    std::optional<FramesPerSecond> m_maxClientPreferredFramesPerSecond;

    Lock m_lock;
    bool m_scheduled WTF_GUARDED_BY_LOCK(m_lock) { false };
    bool m_previousFrameDone WTF_GUARDED_BY_LOCK(m_lock) { true };
    
    unsigned m_unscheduledFireCount WTF_GUARDED_BY_LOCK(m_lock) { 0 };
    unsigned m_maxUnscheduledFireCount WTF_GUARDED_BY_LOCK(m_lock) { 0 };
};

}
