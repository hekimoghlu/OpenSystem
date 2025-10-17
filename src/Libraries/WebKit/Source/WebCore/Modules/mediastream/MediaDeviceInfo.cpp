/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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
#include "MediaDeviceInfo.h"

#if ENABLE(MEDIA_STREAM)

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MediaDeviceInfo);

Ref<MediaDeviceInfo> MediaDeviceInfo::create(const String& label, const String& deviceId, const String& groupId, Kind kind)
{
    return adoptRef(*new MediaDeviceInfo(label, deviceId, groupId, kind));
}

MediaDeviceInfo::MediaDeviceInfo(const String& label, const String& deviceId, const String& groupId, Kind kind)
    : m_label(label)
    , m_deviceId(deviceId)
    , m_groupId(groupId)
    , m_kind(kind)
{
}

MediaDeviceInfo::Kind toMediaDeviceInfoKind(CaptureDevice::DeviceType type)
{
    switch (type) {
    case CaptureDevice::DeviceType::Microphone:
        return MediaDeviceInfo::Kind::Audioinput;
    case CaptureDevice::DeviceType::Speaker:
        return MediaDeviceInfo::Kind::Audiooutput;
    case CaptureDevice::DeviceType::Camera:
    case CaptureDevice::DeviceType::Screen:
    case CaptureDevice::DeviceType::Window:
        return MediaDeviceInfo::Kind::Videoinput;
    case CaptureDevice::DeviceType::SystemAudio:
    case CaptureDevice::DeviceType::Unknown:
        break;
    }
    ASSERT_NOT_REACHED();
    return MediaDeviceInfo::Kind::Audioinput;
}

} // namespace WebCore

#endif
