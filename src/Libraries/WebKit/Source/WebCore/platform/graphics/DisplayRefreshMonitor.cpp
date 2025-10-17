/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
#include "DisplayRefreshMonitor.h"

#include "DisplayRefreshMonitorClient.h"
#include "DisplayRefreshMonitorFactory.h"
#include "DisplayRefreshMonitorManager.h"
#include "Logging.h"
#include <wtf/text/TextStream.h>

#if PLATFORM(IOS_FAMILY)
#include "DisplayRefreshMonitorIOS.h"
#elif PLATFORM(MAC)
#include "LegacyDisplayRefreshMonitorMac.h"
#elif PLATFORM(WIN)
#include "DisplayRefreshMonitorWin.h"
#endif

namespace WebCore {

RefPtr<DisplayRefreshMonitor> DisplayRefreshMonitor::createDefaultDisplayRefreshMonitor(PlatformDisplayID displayID)
{
#if PLATFORM(MAC)
    return LegacyDisplayRefreshMonitorMac::create(displayID);
#endif
#if PLATFORM(IOS_FAMILY)
    return DisplayRefreshMonitorIOS::create(displayID);
#endif
#if PLATFORM(WIN)
    return DisplayRefreshMonitorWin::create(displayID);
#endif
    UNUSED_PARAM(displayID);
    return nullptr;
}

RefPtr<DisplayRefreshMonitor> DisplayRefreshMonitor::create(DisplayRefreshMonitorFactory* factory, PlatformDisplayID displayID)
{
    if (factory) {
        auto monitor = factory->createDisplayRefreshMonitor(displayID);
        if (monitor)
            return monitor;
    }
    
    return createDefaultDisplayRefreshMonitor(displayID);
}

DisplayRefreshMonitor::DisplayRefreshMonitor(PlatformDisplayID displayID)
    : m_displayID(displayID)
{
}

DisplayRefreshMonitor::~DisplayRefreshMonitor() = default;

void DisplayRefreshMonitor::stop()
{
    stopNotificationMechanism();

    Locker locker { m_lock };
    setIsScheduled(false);
}

void DisplayRefreshMonitor::addClient(DisplayRefreshMonitorClient& client)
{
    auto addResult = m_clients.add(&client);
    if (addResult.isNewEntry) {
        LOG_WITH_STREAM(DisplayLink, stream << "[Web] DisplayRefreshMonitor " << this << " addedClient - displayID " << m_displayID << " client " << &client << " client preferred fps " << client.preferredFramesPerSecond());
        computeMaxPreferredFramesPerSecond();
    }
}

bool DisplayRefreshMonitor::removeClient(DisplayRefreshMonitorClient& client)
{
    if (m_clientsToBeNotified)
        m_clientsToBeNotified->remove(&client);

    bool removed = m_clients.remove(&client);
    if (removed) {
        LOG_WITH_STREAM(DisplayLink, stream << "[Web] DisplayRefreshMonitor " << this << " removedClient " << &client);
        computeMaxPreferredFramesPerSecond();
    }

    return removed;
}

std::optional<FramesPerSecond> DisplayRefreshMonitor::maximumClientPreferredFramesPerSecond() const
{
    std::optional<FramesPerSecond> maxFramesPerSecond;
    for (auto& client : m_clients)
        maxFramesPerSecond = std::max<FramesPerSecond>(maxFramesPerSecond.value_or(0), client->preferredFramesPerSecond());

    return maxFramesPerSecond;
}

void DisplayRefreshMonitor::computeMaxPreferredFramesPerSecond()
{
    auto maxFramesPerSecond = maximumClientPreferredFramesPerSecond();
    LOG_WITH_STREAM(DisplayLink, stream << "[Web] DisplayRefreshMonitor " << this << " computeMaxPreferredFramesPerSecond - displayID " << m_displayID << " adjusting max fps to " << maxFramesPerSecond);
    if (maxFramesPerSecond != m_maxClientPreferredFramesPerSecond) {
        m_maxClientPreferredFramesPerSecond = maxFramesPerSecond;
        if (m_maxClientPreferredFramesPerSecond)
            adjustPreferredFramesPerSecond(*m_maxClientPreferredFramesPerSecond);
    }
}

void DisplayRefreshMonitor::clientPreferredFramesPerSecondChanged(DisplayRefreshMonitorClient&)
{
    computeMaxPreferredFramesPerSecond();
}

bool DisplayRefreshMonitor::requestRefreshCallback()
{
    Locker locker { m_lock };

    if (isScheduled())
        return true;

    if (!startNotificationMechanism())
        return false;

    setIsScheduled(true);
    return true;
}

bool DisplayRefreshMonitor::firedAndReachedMaxUnscheduledFireCount()
{
    if (isScheduled()) {
        m_unscheduledFireCount = 0;
        return false;
    }

    ++m_unscheduledFireCount;
    return m_unscheduledFireCount > m_maxUnscheduledFireCount;
}

void DisplayRefreshMonitor::displayLinkFired(const DisplayUpdate& displayUpdate)
{
    {
        Locker locker { m_lock };

        // This may be off the main thread.
        if (!isPreviousFrameDone()) {
            RELEASE_LOG(DisplayLink, "[Web] DisplayRefreshMonitor::displayLinkFired for display %u - previous frame is not complete", displayID());
            return;
        }

        LOG_WITH_STREAM(DisplayLink, stream << "[Web] DisplayRefreshMonitor::displayLinkFired for display " << displayID() << " - scheduled " << isScheduled() << " unscheduledFireCount " << m_unscheduledFireCount << " of " << m_maxUnscheduledFireCount);
        if (firedAndReachedMaxUnscheduledFireCount()) {
            stopNotificationMechanism();
            return;
        }

        setIsScheduled(false);
        setIsPreviousFrameDone(false);
    }
    dispatchDisplayDidRefresh(displayUpdate);
}

void DisplayRefreshMonitor::dispatchDisplayDidRefresh(const DisplayUpdate& displayUpdate)
{
    ASSERT(isMainThread());
    displayDidRefresh(displayUpdate);
}

void DisplayRefreshMonitor::displayDidRefresh(const DisplayUpdate& displayUpdate)
{
    ASSERT(isMainThread());

    UNUSED_PARAM(displayUpdate);
    LOG_WITH_STREAM(DisplayLink, stream << "DisplayRefreshMonitor::displayDidRefresh for display " << displayID() << " update " << displayUpdate);

    // The call back can cause all our clients to be unregistered, so we need to protect
    // against deletion until the end of the method.
    Ref<DisplayRefreshMonitor> protectedThis(*this);

    // Copy the hash table and remove clients from it one by one so we don't notify
    // any client twice, but can respond to removal of clients during the delivery process.
    auto clientsToBeNotified = m_clients;
    m_clientsToBeNotified = &clientsToBeNotified;
    while (!clientsToBeNotified.isEmpty()) {
        auto client = clientsToBeNotified.takeAny();
        client->fireDisplayRefreshIfNeeded(displayUpdate);

        // This checks if this function was reentered. In that case, stop iterating
        // since it's not safe to use the set any more.
        if (m_clientsToBeNotified != &clientsToBeNotified)
            break;
    }

    if (m_clientsToBeNotified == &clientsToBeNotified)
        m_clientsToBeNotified = nullptr;

    {
        Locker locker { m_lock };
        setIsPreviousFrameDone(true);
    }

    DisplayRefreshMonitorManager::sharedManager().displayMonitorDisplayDidRefresh(*this);
}

}
