/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#include "UserMediaCaptureManager.h"

#if USE(GLIB) && ENABLE(MEDIA_STREAM)

#include "UserMediaCaptureManagerMessages.h"
#include "WebProcess.h"
#include <WebCore/CaptureDeviceWithCapabilities.h>
#include <WebCore/MediaDeviceHashSalts.h>
#include <WebCore/MediaStreamRequest.h>
#include <WebCore/RealtimeMediaSourceCenter.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(UserMediaCaptureManager);

UserMediaCaptureManager::UserMediaCaptureManager(WebProcess& process)
    : m_process(process)
{
    process.addMessageReceiver(Messages::UserMediaCaptureManager::messageReceiverName(), *this);
}

UserMediaCaptureManager::~UserMediaCaptureManager()
{
    m_process->removeMessageReceiver(Messages::UserMediaCaptureManager::messageReceiverName());
}

void UserMediaCaptureManager::ref() const
{
    m_process->ref();
}

void UserMediaCaptureManager::deref() const
{
    m_process->deref();
}

void UserMediaCaptureManager::validateUserMediaRequestConstraints(WebCore::MediaStreamRequest request, WebCore::MediaDeviceHashSalts&& deviceIdentifierHashSalts, ValidateUserMediaRequestConstraintsCallback&& completionHandler)
{
    m_validateUserMediaRequestConstraintsCallback = WTFMove(completionHandler);
    auto invalidHandler = [this](auto invalidConstraint) mutable {
        Vector<CaptureDevice> audioDevices;
        Vector<CaptureDevice> videoDevices;
        m_validateUserMediaRequestConstraintsCallback(invalidConstraint, audioDevices, videoDevices);
    };

    auto validHandler = [this](Vector<CaptureDevice>&& audioDevices, Vector<CaptureDevice>&& videoDevices) mutable {
        m_validateUserMediaRequestConstraintsCallback(std::nullopt, audioDevices, videoDevices);
    };

    RealtimeMediaSourceCenter::singleton().validateRequestConstraints(WTFMove(validHandler), WTFMove(invalidHandler), request, WTFMove(deviceIdentifierHashSalts));
}

void UserMediaCaptureManager::getMediaStreamDevices(bool revealIdsAndLabels, GetMediaStreamDevicesCallback&& completionHandler)
{
    RealtimeMediaSourceCenter::singleton().getMediaStreamDevices([completionHandler = WTFMove(completionHandler), revealIdsAndLabels](auto&& devices) mutable {
        auto devicesWithCapabilities = WTF::compactMap(devices, [&](auto& device) -> std::optional<CaptureDeviceWithCapabilities> {
            RealtimeMediaSourceCapabilities deviceCapabilities;

            if (device.isInputDevice()) {
                auto capabilities = RealtimeMediaSourceCenter::singleton().getCapabilities(device);
                if (!capabilities)
                    return std::nullopt;

                if (revealIdsAndLabels)
                    deviceCapabilities = *capabilities;
            }

            return CaptureDeviceWithCapabilities { WTFMove(device), WTFMove(deviceCapabilities) };
        });

        completionHandler(WTFMove(devicesWithCapabilities));
    });
}

} // namespace WebKit

#endif
