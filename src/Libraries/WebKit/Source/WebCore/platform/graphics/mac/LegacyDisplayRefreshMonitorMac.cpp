/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#include "LegacyDisplayRefreshMonitorMac.h"

#if PLATFORM(MAC)

#include "Logging.h"
#include <CoreVideo/CVDisplayLink.h>
#include <wtf/RunLoop.h>
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

constexpr unsigned maxUnscheduledFireCount { 20 };

LegacyDisplayRefreshMonitorMac::LegacyDisplayRefreshMonitorMac(PlatformDisplayID displayID)
    : DisplayRefreshMonitor(displayID)
{
    ASSERT(!isInWebProcess());
    setMaxUnscheduledFireCount(maxUnscheduledFireCount);
}

LegacyDisplayRefreshMonitorMac::~LegacyDisplayRefreshMonitorMac()
{
    ASSERT(!m_displayLink);
}

void LegacyDisplayRefreshMonitorMac::stop()
{
    DisplayRefreshMonitor::stop();
    LOG_WITH_STREAM(DisplayLink, stream << "LegacyDisplayRefreshMonitorMac::stop for dipslay " << displayID() << " destroying display link");
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CVDisplayLinkRelease(m_displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
    m_displayLink = nullptr;
}

static CVReturn displayLinkCallback(CVDisplayLinkRef, const CVTimeStamp*, const CVTimeStamp*, CVOptionFlags, CVOptionFlags*, void* data)
{
    LegacyDisplayRefreshMonitorMac* monitor = static_cast<LegacyDisplayRefreshMonitorMac*>(data);
    monitor->displayLinkCallbackFired();
    return kCVReturnSuccess;
}

void LegacyDisplayRefreshMonitorMac::displayLinkCallbackFired()
{
    displayLinkFired(m_currentUpdate);
    m_currentUpdate = m_currentUpdate.nextUpdate();
}

void LegacyDisplayRefreshMonitorMac::dispatchDisplayDidRefresh(const DisplayUpdate& displayUpdate)
{
    RunLoop::main().dispatch([this, displayUpdate, protectedThis = Ref { *this }] {
        if (m_displayLink)
            displayDidRefresh(displayUpdate);
    });
}

WebCore::FramesPerSecond LegacyDisplayRefreshMonitorMac::nominalFramesPerSecondFromDisplayLink(CVDisplayLinkRef displayLink)
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CVTime refreshPeriod = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
    return round((double)refreshPeriod.timeScale / (double)refreshPeriod.timeValue);
}

bool LegacyDisplayRefreshMonitorMac::ensureDisplayLink()
{
    if (m_displayLink)
        return true;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    auto error = CVDisplayLinkCreateWithCGDisplay(displayID(), &m_displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (error)
        return false;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    error = CVDisplayLinkSetOutputCallback(m_displayLink, displayLinkCallback, this);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (error)
        return false;
        
    return true;
}

bool LegacyDisplayRefreshMonitorMac::startNotificationMechanism()
{
    if (!m_displayLink) {
        if (!ensureDisplayLink())
            return false;
    }

    if (!m_displayLinkIsActive) {
        LOG_WITH_STREAM(DisplayLink, stream << "LegacyDisplayRefreshMonitorMac::startNotificationMechanism for display " << displayID() << " starting display link");

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        auto error = CVDisplayLinkStart(m_displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
        if (error)
            return false;
        
        m_displayLinkIsActive = true;
        m_currentUpdate = { 0, nominalFramesPerSecondFromDisplayLink(m_displayLink) };
    }

    return true;
}

void LegacyDisplayRefreshMonitorMac::stopNotificationMechanism()
{
    if (!m_displayLinkIsActive)
        return;

    if (m_displayLink) {
        LOG_WITH_STREAM(DisplayLink, stream << "LegacyDisplayRefreshMonitorMac::stopNotificationMechanism for display " << displayID() << " stopping display link");
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        CVDisplayLinkStop(m_displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
    }
        
    m_displayLinkIsActive = false;
}

std::optional<FramesPerSecond> LegacyDisplayRefreshMonitorMac::displayNominalFramesPerSecond()
{
    if (!ensureDisplayLink())
        return std::nullopt;
        
    return nominalFramesPerSecondFromDisplayLink(m_displayLink);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
