/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

#include <WebCore/DisplayRefreshMonitor.h>

#if USE(COORDINATED_GRAPHICS)
#include <wtf/RunLoop.h>

namespace WebKit {

class ThreadedCompositor;

class ThreadedDisplayRefreshMonitor : public WebCore::DisplayRefreshMonitor {
public:
    class Client {
    public:
        virtual void requestDisplayRefreshMonitorUpdate() = 0;
        virtual void handleDisplayRefreshMonitorUpdate(bool) = 0;
    };

    static Ref<ThreadedDisplayRefreshMonitor> create(WebCore::PlatformDisplayID displayID, Client& client, WebCore::DisplayUpdate displayUpdate)
    {
        return adoptRef(*new ThreadedDisplayRefreshMonitor(displayID, client, displayUpdate));
    }
    virtual ~ThreadedDisplayRefreshMonitor() = default;

    bool requestRefreshCallback() override;

    bool requiresDisplayRefreshCallback(const WebCore::DisplayUpdate&);
    void dispatchDisplayRefreshCallback();
    void invalidate();

private:
    ThreadedDisplayRefreshMonitor(WebCore::PlatformDisplayID, Client&, WebCore::DisplayUpdate);

    bool startNotificationMechanism() final { return true; }
    void stopNotificationMechanism() final { }

    void displayRefreshCallback();
    RunLoop::Timer m_displayRefreshTimer;
    Client* m_client;
    WebCore::DisplayUpdate m_displayUpdate;
};

} // namespace WebKit

#endif // USE(COORDINATED_GRAPHICS)
