/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "UserMediaPermissionRequestProxy.h"
#include "WebPageProxy.h"
#include <WebCore/SecurityOriginData.h>
#include <wtf/CompletionHandler.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SecurityOrigin;
}

namespace WebKit {

class DisplayCaptureSessionManager {
public:
    static DisplayCaptureSessionManager& singleton();
    static bool isAvailable();

    DisplayCaptureSessionManager();
    ~DisplayCaptureSessionManager();

    void promptForGetDisplayMedia(UserMediaPermissionRequestProxy::UserMediaDisplayCapturePromptType, WebPageProxy&, const WebCore::SecurityOriginData&, CompletionHandler<void(std::optional<WebCore::CaptureDevice>)>&&);
    void cancelGetDisplayMediaPrompt(WebPageProxy&);
    bool canRequestDisplayCapturePermission();
    void setIndexOfDeviceSelectedForTesting(std::optional<unsigned> index) { m_indexOfDeviceSelectedForTesting = index; }

    enum class PromptOverride { Default, CanPrompt, CanNotPrompt };
    void setSystemCanPromptForTesting(bool canPrompt) { m_systemCanPromptForTesting = canPrompt ? PromptOverride::CanPrompt : PromptOverride::CanNotPrompt; }
    bool overrideCanRequestDisplayCapturePermissionForTesting() const { return useMockCaptureDevices() && m_systemCanPromptForTesting != PromptOverride::Default; }

private:

#if HAVE(SCREEN_CAPTURE_KIT)
    enum class CaptureSessionType { None, Screen, Window };
    void alertForGetDisplayMedia(WebPageProxy&, const WebCore::SecurityOriginData&, CompletionHandler<void(DisplayCaptureSessionManager::CaptureSessionType)>&&);
#endif
    void showWindowPicker(const WebCore::SecurityOriginData&, CompletionHandler<void(std::optional<WebCore::CaptureDevice>)>&&);
    void showScreenPicker(const WebCore::SecurityOriginData&, CompletionHandler<void(std::optional<WebCore::CaptureDevice>)>&&);
    std::optional<WebCore::CaptureDevice> deviceSelectedForTesting(WebCore::CaptureDevice::DeviceType, unsigned);

    bool useMockCaptureDevices() const;

    std::optional<unsigned> m_indexOfDeviceSelectedForTesting;
    PromptOverride m_systemCanPromptForTesting { PromptOverride::Default };
};

} // namespace WebKit

#endif // PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
