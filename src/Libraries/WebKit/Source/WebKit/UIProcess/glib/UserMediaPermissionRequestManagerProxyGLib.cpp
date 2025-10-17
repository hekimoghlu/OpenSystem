/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
#include "UserMediaPermissionRequestManagerProxy.h"

#include "UserMediaCaptureManagerMessages.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"
#include <WebCore/CaptureDeviceWithCapabilities.h>
#include <WebCore/MediaConstraintType.h>
#include <WebCore/UserMediaRequest.h>

namespace WebKit {
using namespace WebCore;

void UserMediaPermissionRequestManagerProxy::platformValidateUserMediaRequestConstraints(RealtimeMediaSourceCenter::ValidConstraintsHandler&& validHandler, RealtimeMediaSourceCenter::InvalidConstraintsHandler&& invalidHandler, WebCore::MediaDeviceHashSalts&& deviceIDHashSalts)
{
    m_page->legacyMainFrameProcess().protectedConnection()->sendWithAsyncReply(Messages::UserMediaCaptureManager::ValidateUserMediaRequestConstraints(m_currentUserMediaRequest->userRequest(), WTFMove(deviceIDHashSalts)), [validHandler = WTFMove(validHandler), invalidHandler = WTFMove(invalidHandler)](std::optional<WebCore::MediaConstraintType> invalidConstraint, Vector<WebCore::CaptureDevice> audioDevices, Vector<WebCore::CaptureDevice> videoDevices) mutable {
        if (invalidConstraint)
            invalidHandler(*invalidConstraint);
        else
            validHandler(WTFMove(audioDevices), WTFMove(videoDevices));
    });
}

void UserMediaPermissionRequestManagerProxy::platformGetMediaStreamDevices(bool revealIdsAndLabels, CompletionHandler<void(Vector<CaptureDeviceWithCapabilities>&&)>&& completionHandler)
{
    m_page->legacyMainFrameProcess().protectedConnection()->sendWithAsyncReply(Messages::UserMediaCaptureManager::GetMediaStreamDevices(revealIdsAndLabels), WTFMove(completionHandler));
}

} // namespace WebKit
