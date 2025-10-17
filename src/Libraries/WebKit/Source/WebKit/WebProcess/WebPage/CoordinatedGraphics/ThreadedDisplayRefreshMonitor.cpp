/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
#include "ThreadedDisplayRefreshMonitor.h"

#if USE(COORDINATED_GRAPHICS)

#include "CompositingRunLoop.h"
#include "ThreadedCompositor.h"

#if USE(GLIB_EVENT_LOOP)
#include <wtf/glib/RunLoopSourcePriority.h>
#endif

namespace WebKit {

ThreadedDisplayRefreshMonitor::ThreadedDisplayRefreshMonitor(WebCore::PlatformDisplayID displayID, Client& client, WebCore::DisplayUpdate displayUpdate)
    : WebCore::DisplayRefreshMonitor(displayID)
    , m_displayRefreshTimer(RunLoop::main(), this, &ThreadedDisplayRefreshMonitor::displayRefreshCallback)
    , m_client(&client)
    , m_displayUpdate(displayUpdate)
{
#if USE(GLIB_EVENT_LOOP)
    m_displayRefreshTimer.setPriority(RunLoopSourcePriority::DisplayRefreshMonitorTimer);
    m_displayRefreshTimer.setName("[WebKit] ThreadedDisplayRefreshMonitor");
#endif
}

bool ThreadedDisplayRefreshMonitor::requestRefreshCallback()
{
    if (!m_client)
        return false;

    bool previousFrameDone { false };
    {
        Locker locker { lock() };
        setIsScheduled(true);
        previousFrameDone = isPreviousFrameDone();
    }

    // Only request an update in case we're not currently handling the display
    // refresh notifications under ThreadedDisplayRefreshMonitor::displayRefreshCallback().
    // Any such schedule request is handled in that method after the notifications.
    if (previousFrameDone)
        m_client->requestDisplayRefreshMonitorUpdate();

    return true;
}

bool ThreadedDisplayRefreshMonitor::requiresDisplayRefreshCallback(const WebCore::DisplayUpdate& displayUpdate)
{
    Locker locker { lock() };
    m_displayUpdate = displayUpdate;
    return isScheduled() && isPreviousFrameDone();
}

void ThreadedDisplayRefreshMonitor::dispatchDisplayRefreshCallback()
{
    if (!m_client)
        return;
    m_displayRefreshTimer.startOneShot(0_s);
}

void ThreadedDisplayRefreshMonitor::invalidate()
{
    m_displayRefreshTimer.stop();
    bool wasScheduled = false;
    {
        Locker locker { lock() };
        wasScheduled = isScheduled();
    }
    if (wasScheduled) {
        // This is shutting down, so there's no up-to-date DisplayUpdate available.
        // Instead, the current value is progressed and used for this dispatch.
        m_displayUpdate = m_displayUpdate.nextUpdate();
        displayDidRefresh(m_displayUpdate);
    }
    m_client = nullptr;
}

// FIXME: Refactor to share more code with DisplayRefreshMonitor::displayLinkFired().
void ThreadedDisplayRefreshMonitor::displayRefreshCallback()
{
    bool shouldHandleDisplayRefreshNotification { false };
    WebCore::DisplayUpdate displayUpdate;
    {
        Locker locker { lock() };
        shouldHandleDisplayRefreshNotification = isScheduled() && isPreviousFrameDone();
        displayUpdate = m_displayUpdate;
        if (shouldHandleDisplayRefreshNotification) {
            setIsScheduled(false);
            setIsPreviousFrameDone(false);
        }
    }

    if (shouldHandleDisplayRefreshNotification)
        displayDidRefresh(displayUpdate);

    // Retrieve the scheduled status for this DisplayRefreshMonitor.
    bool hasBeenRescheduled { false };
    {
        Locker locker { lock() };
        hasBeenRescheduled = isScheduled();
    }

    // Notify the compositor about the completed DisplayRefreshMonitor update, passing
    // along information about any schedule request that might have occurred during
    // the notification handling.
    if (m_client)
        m_client->handleDisplayRefreshMonitorUpdate(hasBeenRescheduled);
}

} // namespace WebKit

#endif // USE(COORDINATED_GRAPHICS)
