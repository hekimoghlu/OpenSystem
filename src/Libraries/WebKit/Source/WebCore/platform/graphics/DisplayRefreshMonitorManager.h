/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#include "DisplayRefreshMonitor.h"
#include "PlatformScreen.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

struct DisplayUpdate;

class DisplayRefreshMonitorManager {
    friend class NeverDestroyed<DisplayRefreshMonitorManager>;
    friend class DisplayRefreshMonitor;
public:
    WEBCORE_EXPORT static DisplayRefreshMonitorManager& sharedManager();

    void unregisterClient(DisplayRefreshMonitorClient&);

    bool scheduleAnimation(DisplayRefreshMonitorClient&);
    void windowScreenDidChange(PlatformDisplayID, DisplayRefreshMonitorClient&);
    
    WEBCORE_EXPORT std::optional<FramesPerSecond> nominalFramesPerSecondForDisplay(PlatformDisplayID, DisplayRefreshMonitorFactory*);

    void clientPreferredFramesPerSecondChanged(DisplayRefreshMonitorClient&);

    WEBCORE_EXPORT void displayDidRefresh(PlatformDisplayID, const DisplayUpdate&);

private:
    DisplayRefreshMonitorManager() = default;
    virtual ~DisplayRefreshMonitorManager();

    void displayMonitorDisplayDidRefresh(DisplayRefreshMonitor&);

    size_t findMonitorForDisplayID(PlatformDisplayID) const;
    DisplayRefreshMonitor* monitorForDisplayID(PlatformDisplayID) const;
    DisplayRefreshMonitor* monitorForClient(DisplayRefreshMonitorClient&);

    DisplayRefreshMonitor* ensureMonitorForDisplayID(PlatformDisplayID, DisplayRefreshMonitorFactory*);

    struct DisplayRefreshMonitorWrapper {
        ~DisplayRefreshMonitorWrapper()
        {
            if (monitor)
                monitor->stop();
        }

        RefPtr<DisplayRefreshMonitor> monitor;
    };

    Vector<DisplayRefreshMonitorWrapper> m_monitors;
};

}
