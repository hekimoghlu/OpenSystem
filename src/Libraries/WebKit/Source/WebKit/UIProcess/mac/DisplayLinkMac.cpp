/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
#include "DisplayLink.h"

#if HAVE(DISPLAY_LINK)

#include "Logging.h"
#include <wtf/ProcessPrivilege.h>
#include <wtf/text/TextStream.h>

namespace WebKit {

using namespace WebCore;

void DisplayLink::platformInitialize()
{
    // FIXME: We can get here with displayID == 0 (webkit.org/b/212120), in which case CVDisplayLinkCreateWithCGDisplay()
    // probably defaults to the main screen.
    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanCommunicateWithWindowServer));
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CVReturn error = CVDisplayLinkCreateWithCGDisplay(m_displayID, &m_displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (error) {
        RELEASE_LOG_FAULT(DisplayLink, "Could not create a display link for display %u: error %d", m_displayID, error);
        return;
    }

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    error = CVDisplayLinkSetOutputCallback(m_displayLink, displayLinkCallback, this);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (error) {
        RELEASE_LOG_FAULT(DisplayLink, "DisplayLink: Could not set the display link output callback for display %u: error %d", m_displayID, error);
        return;
    }

    m_displayNominalFramesPerSecond = nominalFramesPerSecondFromDisplayLink(m_displayLink);
}

void DisplayLink::platformFinalize()
{
    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanCommunicateWithWindowServer));
    ASSERT(m_displayLink);
    if (!m_displayLink)
        return;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CVDisplayLinkStop(m_displayLink);
    CVDisplayLinkRelease(m_displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
}

FramesPerSecond DisplayLink::nominalFramesPerSecondFromDisplayLink(CVDisplayLinkRef displayLink)
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CVTime refreshPeriod = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (!refreshPeriod.timeValue)
        return FullSpeedFramesPerSecond;

    FramesPerSecond result = round((double)refreshPeriod.timeScale / (double)refreshPeriod.timeValue);
    return result ?: FullSpeedFramesPerSecond;
}

bool DisplayLink::platformIsRunning() const
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return CVDisplayLinkIsRunning(m_displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
}

void DisplayLink::platformStart()
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CVReturn error = CVDisplayLinkStart(m_displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (error)
        RELEASE_LOG_FAULT(DisplayLink, "DisplayLink: Could not start the display link: %d", error);
}

void DisplayLink::platformStop()
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    CVDisplayLinkStop(m_displayLink);
ALLOW_DEPRECATED_DECLARATIONS_END
}

CVReturn DisplayLink::displayLinkCallback(CVDisplayLinkRef displayLinkRef, const CVTimeStamp*, const CVTimeStamp*, CVOptionFlags, CVOptionFlags*, void* data)
{
    static_cast<DisplayLink*>(data)->notifyObserversDisplayDidRefresh();
    return kCVReturnSuccess;
}

} // namespace WebKit

#endif // HAVE(DISPLAY_LINK)
