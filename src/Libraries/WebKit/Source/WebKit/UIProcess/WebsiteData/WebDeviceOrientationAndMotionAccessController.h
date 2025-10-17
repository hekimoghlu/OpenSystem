/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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

#if ENABLE(DEVICE_ORIENTATION)

#include <WebCore/DeviceOrientationOrMotionPermissionState.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/CanMakeWeakPtr.h>
#include <wtf/HashMap.h>

namespace WebKit {

class WebsiteDataStore;
class WebPageProxy;
class WebFrameProxy;
struct FrameInfoData;

class WebDeviceOrientationAndMotionAccessController : public CanMakeWeakPtr<WebDeviceOrientationAndMotionAccessController> {
public:
    WebDeviceOrientationAndMotionAccessController(WebsiteDataStore&);

    void ref() const;
    void deref() const;

    void shouldAllowAccess(WebPageProxy&, WebFrameProxy&, FrameInfoData&&, bool mayPrompt, CompletionHandler<void(WebCore::DeviceOrientationOrMotionPermissionState)>&&);
    void clearPermissions() { m_deviceOrientationPermissionDecisions.clear(); }

    WebCore::DeviceOrientationOrMotionPermissionState cachedDeviceOrientationPermission(const WebCore::SecurityOriginData&) const;

private:
    HashMap<WebCore::SecurityOriginData, bool> m_deviceOrientationPermissionDecisions;
    HashMap<WebCore::SecurityOriginData, Vector<CompletionHandler<void(WebCore::DeviceOrientationOrMotionPermissionState)>>> m_pendingRequests;

    WeakRef<WebsiteDataStore> m_websiteDataStore;
};

}

#endif
