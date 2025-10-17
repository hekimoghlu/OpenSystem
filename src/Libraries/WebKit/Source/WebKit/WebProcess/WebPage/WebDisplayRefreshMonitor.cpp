/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#include "WebDisplayRefreshMonitor.h"

#if HAVE(DISPLAY_LINK)

#include "Logging.h"
#include "WebProcess.h"
#include "WebProcessProxy.h"
#include "WebProcessProxyMessages.h"
#include <WebCore/AnimationFrameRate.h>
#include <WebCore/DisplayRefreshMonitor.h>
#include <wtf/text/TextStream.h>

#if PLATFORM(MAC)
#include <WebCore/RunLoopObserver.h>
#endif

namespace WebKit {
using namespace WebCore;

// Avoid repeated start/stop IPC when rescheduled inside the callback.
constexpr unsigned maxUnscheduledFireCount { 1 };

WebDisplayRefreshMonitor::WebDisplayRefreshMonitor(PlatformDisplayID displayID)
    : DisplayRefreshMonitor(displayID)
    , m_observerID(DisplayLinkObserverID::generate())
{
    ASSERT(isMainRunLoop());
    setMaxUnscheduledFireCount(maxUnscheduledFireCount);
}

WebDisplayRefreshMonitor::~WebDisplayRefreshMonitor()
{
    // stop() should have been called.
    ASSERT(!m_displayLinkIsActive);
}

#if PLATFORM(MAC)
void WebDisplayRefreshMonitor::dispatchDisplayDidRefresh(const DisplayUpdate& displayUpdate)
{
    // FIXME: This will perturb displayUpdate.
    if (!m_firstCallbackInCurrentRunloop) {
        RELEASE_LOG(DisplayLink, "[Web] WebDisplayRefreshMonitor::dispatchDisplayDidRefresh() for display %u - m_firstCallbackInCurrentRunloop is false", displayID());
        Locker locker { lock() };
        setIsPreviousFrameDone(true);
        return;
    }

    DisplayRefreshMonitor::dispatchDisplayDidRefresh(displayUpdate);
}
#endif

bool WebDisplayRefreshMonitor::startNotificationMechanism()
{
    if (m_displayLinkIsActive)
        return true;

    LOG_WITH_STREAM(DisplayLink, stream << "[Web] WebDisplayRefreshMonitor::requestRefreshCallback for display " << displayID() << " - starting");
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebProcessProxy::StartDisplayLink(m_observerID, displayID(), maxClientPreferredFramesPerSecond().value_or(FullSpeedFramesPerSecond)), 0);

#if PLATFORM(MAC)
    if (!m_runLoopObserver) {
        // The RunLoopObserver repeats.
        // FIXME: Double check whether the value of `DisplayRefreshMonitor` (1) is the appropriate runloop order here,
        // and also whether we should be specifying `RunLoopObserver::Activity::Entry` when scheduling the observer below.
        m_runLoopObserver = makeUnique<RunLoopObserver>(RunLoopObserver::WellKnownOrder::DisplayRefreshMonitor, [this] {
            m_firstCallbackInCurrentRunloop = true;
        });
    }

    m_runLoopObserver->schedule(CFRunLoopGetCurrent());
#endif
    m_displayLinkIsActive = true;

    return true;
}

void WebDisplayRefreshMonitor::stopNotificationMechanism()
{
    if (!m_displayLinkIsActive)
        return;

    LOG_WITH_STREAM(DisplayLink, stream << "[Web] WebDisplayRefreshMonitor::requestRefreshCallback - stopping");
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebProcessProxy::StopDisplayLink(m_observerID, displayID()), 0);
#if PLATFORM(MAC)
    m_runLoopObserver->invalidate();
#endif
    m_displayLinkIsActive = false;
}

void WebDisplayRefreshMonitor::adjustPreferredFramesPerSecond(FramesPerSecond preferredFramesPerSecond)
{
    LOG_WITH_STREAM(DisplayLink, stream << "[Web] WebDisplayRefreshMonitor::adjustPreferredFramesPerSecond for display link on display " << displayID() << " to " << preferredFramesPerSecond);
    WebProcess::singleton().parentProcessConnection()->send(Messages::WebProcessProxy::SetDisplayLinkPreferredFramesPerSecond(m_observerID, displayID(), preferredFramesPerSecond), 0);
}

} // namespace WebKit

#endif // HAVE(DISPLAY_LINK)
