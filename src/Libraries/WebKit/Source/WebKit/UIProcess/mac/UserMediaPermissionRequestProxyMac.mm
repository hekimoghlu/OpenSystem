/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
#import "config.h"
#import "UserMediaPermissionRequestProxyMac.h"

#import "DisplayCaptureSessionManager.h"
#import "UserMediaPermissionRequestManagerProxy.h"
#import "WebPreferences.h"

namespace WebKit {
using namespace WebCore;

Ref<UserMediaPermissionRequestProxy> UserMediaPermissionRequestProxy::create(UserMediaPermissionRequestManagerProxy& manager, std::optional<UserMediaRequestIdentifier> userMediaID, FrameIdentifier mainFrameID, FrameIdentifier frameID, Ref<SecurityOrigin>&& userMediaDocumentOrigin, Ref<SecurityOrigin>&& topLevelDocumentOrigin, Vector<CaptureDevice>&& audioDevices, Vector<CaptureDevice>&& videoDevices, MediaStreamRequest&& request, CompletionHandler<void(bool)>&& decisionCompletionHandler)
{
    return adoptRef(*new UserMediaPermissionRequestProxyMac(manager, userMediaID, mainFrameID, frameID, WTFMove(userMediaDocumentOrigin), WTFMove(topLevelDocumentOrigin), WTFMove(audioDevices), WTFMove(videoDevices), WTFMove(request), WTFMove(decisionCompletionHandler)));
}

UserMediaPermissionRequestProxyMac::UserMediaPermissionRequestProxyMac(UserMediaPermissionRequestManagerProxy& manager, std::optional<UserMediaRequestIdentifier> userMediaID, FrameIdentifier mainFrameID, FrameIdentifier frameID, Ref<SecurityOrigin>&& userMediaDocumentOrigin, Ref<SecurityOrigin>&& topLevelDocumentOrigin, Vector<CaptureDevice>&& audioDevices, Vector<CaptureDevice>&& videoDevices, MediaStreamRequest&& request, CompletionHandler<void(bool)>&& decisionCompletionHandler)
    : UserMediaPermissionRequestProxy(manager, userMediaID, mainFrameID, frameID, WTFMove(userMediaDocumentOrigin), WTFMove(topLevelDocumentOrigin), WTFMove(audioDevices), WTFMove(videoDevices), WTFMove(request), WTFMove(decisionCompletionHandler))
{
}

UserMediaPermissionRequestProxyMac::~UserMediaPermissionRequestProxyMac()
{
}

void UserMediaPermissionRequestProxyMac::invalidate()
{
#if ENABLE(MEDIA_STREAM)
    if (m_hasPendingGetDispayMediaPrompt) {
        if (RefPtr page = protectedManager()->page())
            DisplayCaptureSessionManager::singleton().cancelGetDisplayMediaPrompt(*page);
        m_hasPendingGetDispayMediaPrompt = false;
    }
#endif
    UserMediaPermissionRequestProxy::invalidate();
}

void UserMediaPermissionRequestProxyMac::promptForGetDisplayMedia(UserMediaDisplayCapturePromptType promptType)
{
#if ENABLE(MEDIA_STREAM)
    if (!manager())
        return;

    RefPtr page = protectedManager()->page();
    if (!page)
        return;

    m_hasPendingGetDispayMediaPrompt = true;
    DisplayCaptureSessionManager::singleton().promptForGetDisplayMedia(promptType, *page, topLevelDocumentSecurityOrigin().data(), [protectedThis = Ref { *this }](std::optional<CaptureDevice> device) mutable {

        protectedThis->m_hasPendingGetDispayMediaPrompt = false;

        if (!device) {
            protectedThis->deny(UserMediaPermissionRequestProxy::UserMediaAccessDenialReason::PermissionDenied);
            return;
        }

        protectedThis->setEligibleVideoDeviceUIDs({ device.value() });
        protectedThis->allow(String(), device.value().persistentId());
    });
#else
    ASSERT_NOT_REACHED();
#endif
}

bool UserMediaPermissionRequestProxyMac::canRequestDisplayCapturePermission()
{
#if ENABLE(MEDIA_STREAM)
    auto overridePreference = DisplayCaptureSessionManager::singleton().overrideCanRequestDisplayCapturePermissionForTesting();
    RefPtr manager = this->manager();
    if (!manager || !manager->page() || (!overridePreference && manager->page()->preferences().requireUAGetDisplayMediaPrompt()))
        return false;

    return DisplayCaptureSessionManager::singleton().canRequestDisplayCapturePermission();
#else
    return false;
#endif
}

} // namespace WebKit
