/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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
#include "WKMockMediaDevice.h"

#include "WKAPICast.h"
#include "WKDictionary.h"
#include "WKRetainPtr.h"
#include "WKString.h"
#include "WebProcessPool.h"
#include <WebCore/MockMediaDevice.h>

using namespace WebKit;

void WKAddMockMediaDevice(WKContextRef context, WKStringRef persistentId, WKStringRef label, WKStringRef type, WKDictionaryRef properties)
{
#if ENABLE(MEDIA_STREAM)
    String typeString = WebKit::toImpl(type)->string();
    std::variant<WebCore::MockMicrophoneProperties, WebCore::MockSpeakerProperties, WebCore::MockCameraProperties, WebCore::MockDisplayProperties> deviceProperties;
    if (typeString == "camera"_s) {
        WebCore::MockCameraProperties cameraProperties;
        if (properties) {
            auto facingModeKey = adoptWK(WKStringCreateWithUTF8CString("facingMode"));
            if (auto facingMode = WKDictionaryGetItemForKey(properties, facingModeKey.get())) {
                if (WKStringIsEqualToUTF8CString(static_cast<WKStringRef>(facingMode), "unknown"))
                    cameraProperties.facingMode = WebCore::VideoFacingMode::Unknown;
            }
            auto fillColorKey = adoptWK(WKStringCreateWithUTF8CString("fillColor"));
            if (auto fillColor = WKDictionaryGetItemForKey(properties, fillColorKey.get())) {
                if (WKStringIsEqualToUTF8CString(static_cast<WKStringRef>(fillColor), "green"))
                    cameraProperties.fillColor = WebCore::Color::green;
            }
        }
        deviceProperties = WTFMove(cameraProperties);
    } else if (typeString == "screen"_s)
        deviceProperties = WebCore::MockDisplayProperties { };
    else if (typeString == "speaker"_s)
        deviceProperties = WebCore::MockSpeakerProperties { };
    else if (typeString != "microphone"_s)
        return;

    WebCore::MockMediaDevice::Flags flags;
    if (properties) {
        auto invalidKey = adoptWK(WKStringCreateWithUTF8CString("invalid"));
        if (auto invalid = WKDictionaryGetItemForKey(properties, invalidKey.get())) {
            if (WKStringIsEqualToUTF8CString(static_cast<WKStringRef>(invalid), "true"))
                flags.add(WebCore::MockMediaDevice::Flag::Invalid);
        }
    }

    toImpl(context)->addMockMediaDevice({ WebKit::toImpl(persistentId)->string(), WebKit::toImpl(label)->string(), flags, WTFMove(deviceProperties) });
#endif
}

void WKClearMockMediaDevices(WKContextRef context)
{
    toImpl(context)->clearMockMediaDevices();
}

void WKRemoveMockMediaDevice(WKContextRef context, WKStringRef persistentId)
{
    toImpl(context)->removeMockMediaDevice(WebKit::toImpl(persistentId)->string());
}

void WKResetMockMediaDevices(WKContextRef context)
{
    toImpl(context)->resetMockMediaDevices();
}

void WKSetMockMediaDeviceIsEphemeral(WKContextRef context, WKStringRef persistentId, bool isEphemeral)
{
    toImpl(context)->setMockMediaDeviceIsEphemeral(WebKit::toImpl(persistentId)->string(), isEphemeral);
}
