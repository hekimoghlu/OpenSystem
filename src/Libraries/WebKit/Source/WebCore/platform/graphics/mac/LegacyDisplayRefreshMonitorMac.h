/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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

#if PLATFORM(MAC)

#include "DisplayRefreshMonitor.h"
#include <wtf/WeakPtr.h>

typedef struct __CVDisplayLink *CVDisplayLinkRef;

namespace WebCore {

class LegacyDisplayRefreshMonitorMac : public DisplayRefreshMonitor {
public:
    static Ref<LegacyDisplayRefreshMonitorMac> create(PlatformDisplayID displayID)
    {
        return adoptRef(*new LegacyDisplayRefreshMonitorMac(displayID));
    }
    
    virtual ~LegacyDisplayRefreshMonitorMac();

    void displayLinkCallbackFired();

private:
    explicit LegacyDisplayRefreshMonitorMac(PlatformDisplayID);

    void dispatchDisplayDidRefresh(const DisplayUpdate&) final;

    void stop() final;

    bool startNotificationMechanism() final;
    void stopNotificationMechanism() final;
    std::optional<FramesPerSecond> displayNominalFramesPerSecond() final;
    
    bool ensureDisplayLink();

    static FramesPerSecond nominalFramesPerSecondFromDisplayLink(CVDisplayLinkRef);

    CVDisplayLinkRef m_displayLink { nullptr };
    
    DisplayUpdate m_currentUpdate;
    bool m_displayLinkIsActive { false };
};

} // namespace WebCore

#endif // PLATFORM(MAC)
