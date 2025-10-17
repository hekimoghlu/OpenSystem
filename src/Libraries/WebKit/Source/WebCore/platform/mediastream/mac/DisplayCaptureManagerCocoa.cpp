/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#include "DisplayCaptureManagerCocoa.h"
#include "RealtimeMediaSourceCenter.h"

#if ENABLE(MEDIA_STREAM)

#include "Logging.h"
#include <wtf/Algorithms.h>
#include <wtf/NeverDestroyed.h>

#if PLATFORM(IOS) || PLATFORM(VISION)
#include "ReplayKitCaptureSource.h"
#endif

#if HAVE(SCREEN_CAPTURE_KIT)
#include "ScreenCaptureKitCaptureSource.h"
#endif

namespace WebCore {

DisplayCaptureManagerCocoa& DisplayCaptureManagerCocoa::singleton()
{
    static NeverDestroyed<DisplayCaptureManagerCocoa> manager;
    return manager.get();
}

const Vector<CaptureDevice>& DisplayCaptureManagerCocoa::captureDevices()
{
    return m_devices;
}

std::optional<CaptureDevice> DisplayCaptureManagerCocoa::screenCaptureDeviceWithPersistentID(const String& deviceID)
{
#if HAVE(SCREEN_CAPTURE_KIT)
    if (ScreenCaptureKitCaptureSource::isAvailable())
        return ScreenCaptureKitCaptureSource::screenCaptureDeviceWithPersistentID(deviceID);
    ASSERT_NOT_REACHED();
    return std::nullopt;
#else
    UNUSED_PARAM(deviceID);
    return std::nullopt;
#endif
}

std::optional<CaptureDevice> DisplayCaptureManagerCocoa::windowCaptureDeviceWithPersistentID(const String& deviceID)
{
    UNUSED_PARAM(deviceID);

#if HAVE(SCREEN_CAPTURE_KIT)
    if (ScreenCaptureKitCaptureSource::isAvailable())
        return ScreenCaptureKitCaptureSource::windowCaptureDeviceWithPersistentID(deviceID);
#endif

    return std::nullopt;
}

std::optional<CaptureDevice> DisplayCaptureManagerCocoa::captureDeviceWithPersistentID(CaptureDevice::DeviceType type, const String& id)
{
    switch (type) {
    case CaptureDevice::DeviceType::Screen:
        return screenCaptureDeviceWithPersistentID(id);
        break;

    case CaptureDevice::DeviceType::Window:
        return windowCaptureDeviceWithPersistentID(id);
        break;

    case CaptureDevice::DeviceType::SystemAudio:
    case CaptureDevice::DeviceType::Camera:
    case CaptureDevice::DeviceType::Microphone:
    case CaptureDevice::DeviceType::Speaker:
    case CaptureDevice::DeviceType::Unknown:
        ASSERT_NOT_REACHED();
        break;
    }

    return std::nullopt;
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
