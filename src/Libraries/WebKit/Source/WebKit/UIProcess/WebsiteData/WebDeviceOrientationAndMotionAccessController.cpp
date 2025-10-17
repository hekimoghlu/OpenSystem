/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#include "WebDeviceOrientationAndMotionAccessController.h"

#if ENABLE(DEVICE_ORIENTATION)

#include "APIUIClient.h"
#include "FrameInfoData.h"
#include "PageLoadState.h"
#include "WebPageProxy.h"
#include "WebsiteDataStore.h"

namespace WebKit {

using namespace WebCore;

WebDeviceOrientationAndMotionAccessController::WebDeviceOrientationAndMotionAccessController(WebsiteDataStore& websiteDataStore)
    : m_websiteDataStore(websiteDataStore)
{
}

void WebDeviceOrientationAndMotionAccessController::shouldAllowAccess(WebPageProxy& page, WebFrameProxy& frame, FrameInfoData&& frameInfo, bool mayPrompt, CompletionHandler<void(DeviceOrientationOrMotionPermissionState)>&& completionHandler)
{
    auto originData = SecurityOrigin::createFromString(page.pageLoadState().activeURL())->data();
    auto currentPermission = cachedDeviceOrientationPermission(originData);
    if (currentPermission != DeviceOrientationOrMotionPermissionState::Prompt || !mayPrompt)
        return completionHandler(currentPermission);

    auto& pendingRequests = m_pendingRequests.ensure(originData, [] {
        return Vector<CompletionHandler<void(WebCore::DeviceOrientationOrMotionPermissionState)>> { };
    }).iterator->value;
    pendingRequests.append(WTFMove(completionHandler));
    if (pendingRequests.size() > 1)
        return;

    page.uiClient().shouldAllowDeviceOrientationAndMotionAccess(page, frame, WTFMove(frameInfo), [this, weakThis = WeakPtr { *this }, originData](bool granted) mutable {
        if (!weakThis)
            return;
        m_deviceOrientationPermissionDecisions.set(originData, granted);
        auto requests = m_pendingRequests.take(originData);
        for (auto& completionHandler : requests)
            completionHandler(granted ? DeviceOrientationOrMotionPermissionState::Granted : DeviceOrientationOrMotionPermissionState::Denied);
    });
}

DeviceOrientationOrMotionPermissionState WebDeviceOrientationAndMotionAccessController::cachedDeviceOrientationPermission(const SecurityOriginData& origin) const
{
    if (!m_deviceOrientationPermissionDecisions.isValidKey(origin))
        return DeviceOrientationOrMotionPermissionState::Denied;

    auto it = m_deviceOrientationPermissionDecisions.find(origin);
    if (it == m_deviceOrientationPermissionDecisions.end())
        return DeviceOrientationOrMotionPermissionState::Prompt;
    return it->value ? DeviceOrientationOrMotionPermissionState::Granted : DeviceOrientationOrMotionPermissionState::Denied;
}

void WebDeviceOrientationAndMotionAccessController::ref() const
{
    m_websiteDataStore->ref();
}

void WebDeviceOrientationAndMotionAccessController::deref() const
{
    m_websiteDataStore->deref();
}

} // namespace WebKit

#endif // ENABLE(DEVICE_ORIENTATION)
