/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
#include "RenderingUpdateScheduler.h"

#include "Chrome.h"
#include "ChromeClient.h"
#include "DisplayRefreshMonitorManager.h"
#include "Logging.h"
#include "Page.h"
#include "Timer.h"
#include <wtf/SystemTracing.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderingUpdateScheduler);

RenderingUpdateScheduler::RenderingUpdateScheduler(Page& page)
    : m_page(page)
{
    windowScreenDidChange(page.chrome().displayID());
}

RenderingUpdateScheduler::~RenderingUpdateScheduler() = default;

bool RenderingUpdateScheduler::scheduleAnimation()
{
    if (m_useTimer)
        return false;

    return DisplayRefreshMonitorManager::sharedManager().scheduleAnimation(*this);
}

void RenderingUpdateScheduler::adjustRenderingUpdateFrequency()
{
    auto renderingUpdateFramesPerSecond = m_page.preferredRenderingUpdateFramesPerSecond();
    if (renderingUpdateFramesPerSecond) {
        setPreferredFramesPerSecond(renderingUpdateFramesPerSecond.value());
        m_useTimer = false;
    } else
        m_useTimer = true;

    if (m_refreshTimer) {
        clearScheduled();
        scheduleRenderingUpdate();
    }
}

void RenderingUpdateScheduler::scheduleRenderingUpdate()
{
    LOG_WITH_STREAM(EventLoop, stream << "RenderingUpdateScheduler for page " << &m_page << " scheduleTimedRenderingUpdate() - already scheduled " << isScheduled() << " page visible " << m_page.isVisible());

    if (isScheduled())
        return;

    // Optimize the case when an invisible page wants just to schedule layer flush.
    if (!m_page.isVisible()) {
        triggerRenderingUpdate();
        return;
    }

    tracePoint(ScheduleRenderingUpdate);

    if (!scheduleAnimation()) {
        LOG_WITH_STREAM(DisplayLink, stream << "RenderingUpdateScheduler::scheduleRenderingUpdate for interval " << m_page.preferredRenderingUpdateInterval() << " falling back to timer");
        startTimer(m_page.preferredRenderingUpdateInterval());
    }
    
    m_page.didScheduleRenderingUpdate();
}

bool RenderingUpdateScheduler::isScheduled() const
{
    return m_refreshTimer || DisplayRefreshMonitorClient::isScheduled();
}
    
void RenderingUpdateScheduler::startTimer(Seconds delay)
{
    LOG_WITH_STREAM(EventLoop, stream << "RenderingUpdateScheduler for page " << &m_page << " startTimer(" << delay << ")");

    ASSERT(!m_refreshTimer);
    m_refreshTimer = makeUnique<Timer>(*this, &RenderingUpdateScheduler::displayRefreshFired);
    m_refreshTimer->startOneShot(delay);
}

void RenderingUpdateScheduler::clearScheduled()
{
    m_refreshTimer = nullptr;
}

DisplayRefreshMonitorFactory* RenderingUpdateScheduler::displayRefreshMonitorFactory() const
{
    return m_page.chrome().client().displayRefreshMonitorFactory();
}

void RenderingUpdateScheduler::windowScreenDidChange(PlatformDisplayID displayID)
{
    adjustRenderingUpdateFrequency();
    DisplayRefreshMonitorManager::sharedManager().windowScreenDidChange(displayID, *this);
}

void RenderingUpdateScheduler::displayRefreshFired()
{
    LOG_WITH_STREAM(EventLoop, stream << "RenderingUpdateScheduler for page " << &m_page << " displayRefreshFired()");

    tracePoint(TriggerRenderingUpdate);

    clearScheduled();
    
    if (m_page.chrome().client().shouldTriggerRenderingUpdate(m_rescheduledRenderingUpdateCount)) {
        triggerRenderingUpdate();
        m_rescheduledRenderingUpdateCount = 0;
    } else {
        scheduleRenderingUpdate();
        ++m_rescheduledRenderingUpdateCount;
    }
}

void RenderingUpdateScheduler::triggerRenderingUpdate()
{
    m_page.chrome().client().triggerRenderingUpdate();
}

}
