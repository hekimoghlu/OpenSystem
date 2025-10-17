/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

#if ENABLE(MEDIA_STREAM)

#include "UserMediaPermissionRequestManagerProxy.h"
#include <WebCore/CaptureDevice.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/RealtimeMediaSourceCenter.h>
#include <WebCore/UserMediaClient.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class UserMediaPermissionRequestProxy;
class WebProcessProxy;

class UserMediaProcessManager : public WebCore::RealtimeMediaSourceCenterObserver {
public:
    static UserMediaProcessManager& singleton();

    UserMediaProcessManager();

    // No-op since this object is always a singleton.
    void ref() const { ASSERT(this == &singleton()); }
    void deref() const { ASSERT(this == &singleton()); }

    bool willCreateMediaStream(UserMediaPermissionRequestManagerProxy&, const UserMediaPermissionRequestProxy&);

    void revokeSandboxExtensionsIfNeeded(WebProcessProxy&);

    void setCaptureEnabled(bool);
    bool captureEnabled() const { return m_captureEnabled; }

    void denyNextUserMediaRequest() { m_denyNextRequest = true; }

    void beginMonitoringCaptureDevices();

private:

    enum class ShouldNotify : bool { No, Yes };
    void updateCaptureDevices(ShouldNotify);
    void captureDevicesChanged();

    // RealtimeMediaSourceCenterObserver
    void devicesChanged() final;
    void deviceWillBeRemoved(const String& persistentId) final { }

    Vector<WebCore::CaptureDevice> m_captureDevices;
    bool m_captureEnabled { true };
    bool m_denyNextRequest { false };
};

} // namespace WebKit

#endif
