/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#include "DisplayRefreshMonitorManager.h"

#include "DisplayRefreshMonitor.h"
#include "DisplayRefreshMonitorClient.h"
#include "Logging.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

DisplayRefreshMonitorManager::~DisplayRefreshMonitorManager() = default;

DisplayRefreshMonitorManager& DisplayRefreshMonitorManager::sharedManager()
{
    static NeverDestroyed<DisplayRefreshMonitorManager> manager;
    return manager.get();
}

DisplayRefreshMonitor* DisplayRefreshMonitorManager::ensureMonitorForDisplayID(PlatformDisplayID displayID, DisplayRefreshMonitorFactory* factory)
{
    if (auto* existingMonitor = monitorForDisplayID(displayID))
        return existingMonitor;

    auto monitor = DisplayRefreshMonitor::create(factory, displayID);
    if (!monitor)
        return nullptr;

    LOG_WITH_STREAM(DisplayLink, stream << "[Web] DisplayRefreshMonitorManager::ensureMonitorForDisplayID() - created monitor " << monitor.get() << " for display " << displayID);
    DisplayRefreshMonitor* result = monitor.get();
    m_monitors.append(DisplayRefreshMonitorWrapper { WTFMove(monitor) });
    return result;
}

void DisplayRefreshMonitorManager::unregisterClient(DisplayRefreshMonitorClient& client)
{
    if (!client.hasDisplayID())
        return;

    PlatformDisplayID clientDisplayID = client.displayID();
    auto index = findMonitorForDisplayID(clientDisplayID);
    if (index == notFound)
        return;

    RefPtr<DisplayRefreshMonitor> monitor = m_monitors[index].monitor;
    monitor->removeClient(client);
}

void DisplayRefreshMonitorManager::clientPreferredFramesPerSecondChanged(DisplayRefreshMonitorClient& client)
{
    if (RefPtr monitor = monitorForClient(client))
        monitor->clientPreferredFramesPerSecondChanged(client);
}

bool DisplayRefreshMonitorManager::scheduleAnimation(DisplayRefreshMonitorClient& client)
{
    if (RefPtr monitor = monitorForClient(client)) {
        client.setIsScheduled(true);
        return monitor->requestRefreshCallback();
    }
    return false;
}

void DisplayRefreshMonitorManager::displayMonitorDisplayDidRefresh(DisplayRefreshMonitor&)
{
    // Maybe we should remove monitors that haven't been active for some time.
}

void DisplayRefreshMonitorManager::windowScreenDidChange(PlatformDisplayID displayID, DisplayRefreshMonitorClient& client)
{
    if (client.hasDisplayID() && client.displayID() == displayID)
        return;
    
    unregisterClient(client);
    client.setDisplayID(displayID);
    if (client.isScheduled())
        scheduleAnimation(client);
}

std::optional<FramesPerSecond> DisplayRefreshMonitorManager::nominalFramesPerSecondForDisplay(PlatformDisplayID displayID, DisplayRefreshMonitorFactory* factory)
{
    if (RefPtr monitor = ensureMonitorForDisplayID(displayID, factory))
        return monitor->displayNominalFramesPerSecond();

    return std::nullopt;
}

void DisplayRefreshMonitorManager::displayDidRefresh(PlatformDisplayID displayID, const DisplayUpdate& displayUpdate)
{
    if (RefPtr monitor = monitorForDisplayID(displayID))
        monitor->displayLinkFired(displayUpdate);
}

size_t DisplayRefreshMonitorManager::findMonitorForDisplayID(PlatformDisplayID displayID) const
{
    return m_monitors.findIf([&](auto& monitorWrapper) {
        return monitorWrapper.monitor->displayID() == displayID;
    });
}

DisplayRefreshMonitor* DisplayRefreshMonitorManager::monitorForClient(DisplayRefreshMonitorClient& client)
{
    if (!client.hasDisplayID())
        return nullptr;

    RefPtr monitor = ensureMonitorForDisplayID(client.displayID(), client.displayRefreshMonitorFactory());
    if (monitor)
        monitor->addClient(client);

    return monitor.get();
}

DisplayRefreshMonitor* DisplayRefreshMonitorManager::monitorForDisplayID(PlatformDisplayID displayID) const
{
    auto index = findMonitorForDisplayID(displayID);
    return index == notFound ? nullptr : m_monitors[index].monitor.get();
}

}
