/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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

namespace WebKit {

using namespace WebCore;

void DisplayLink::platformInitialize()
{
    // FIXME: We can get here with displayID == 0 (webkit.org/b/212120), in which case DisplayVBlankMonitor defaults to the main screen.
    m_vblankMonitor = DisplayVBlankMonitor::create(m_displayID);
    m_vblankMonitor->setHandler([this] {
        notifyObserversDisplayDidRefresh();
    });

    m_displayNominalFramesPerSecond = m_vblankMonitor->refreshRate();
}

void DisplayLink::platformFinalize()
{
    ASSERT(m_vblankMonitor);
    m_vblankMonitor->invalidate();
}

bool DisplayLink::platformIsRunning() const
{
    return m_vblankMonitor->isActive();
}

void DisplayLink::platformStart()
{
    m_vblankMonitor->start();
}

void DisplayLink::platformStop()
{
    m_vblankMonitor->stop();
}

} // namespace WebKit

