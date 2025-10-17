/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
#include "CoreAudioCaptureDevice.h"

#if ENABLE(MEDIA_STREAM) && PLATFORM(MAC)

#include "CaptureDeviceManager.h"
#include "Logging.h"
#include <AudioUnit/AudioUnit.h>

#import <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

static bool getDeviceInfo(uint32_t deviceID, CaptureDevice::DeviceType type, String& persistentID, String& label)
{
    CFStringRef uniqueID;
    AudioObjectPropertyAddress address {
        kAudioDevicePropertyDeviceUID,
        kAudioDevicePropertyScopeInput,
        kAudioObjectPropertyElementMain
    };
    UInt32 dataSize = sizeof(uniqueID);
    auto err = AudioObjectGetPropertyData(static_cast<UInt32>(deviceID), &address, 0, nullptr, &dataSize, &uniqueID);
    if (err) {
        RELEASE_LOG_ERROR(WebRTC, "CoreAudioCaptureDevice::getDeviceInfo failed to get device unique id with error %d (%.4s)", (int)err, (char*)&err);
        return false;
    }
    persistentID = uniqueID;
    CFRelease(uniqueID);

    CFStringRef localizedName = nullptr;
    AudioObjectPropertyScope scope = type == CaptureDevice::DeviceType::Microphone ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput;
    address = {
        kAudioDevicePropertyDataSource,
        scope,
        kAudioObjectPropertyElementMain
    };
    uint32_t sourceID;
    dataSize = sizeof(sourceID);
    err = AudioObjectGetPropertyData(static_cast<UInt32>(deviceID), &address, 0, nullptr, &dataSize, &sourceID);
    if (!err) {
        AudioValueTranslation translation = { &deviceID, sizeof(deviceID), &localizedName, sizeof(localizedName) };
        address = {
            kAudioDevicePropertyDataSourceNameForIDCFString,
            scope,
            kAudioObjectPropertyElementMain
        };
        dataSize = sizeof(translation);
        err = AudioObjectGetPropertyData(static_cast<UInt32>(deviceID), &address, 0, nullptr, &dataSize, &translation);
    }

    if (err || !localizedName || !CFStringGetLength(localizedName)) {
        address = {
            kAudioObjectPropertyName,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        dataSize = sizeof(localizedName);
        err = AudioObjectGetPropertyData(static_cast<UInt32>(deviceID), &address, 0, nullptr, &dataSize, &localizedName);
    }

    if (err) {
        RELEASE_LOG_ERROR(WebRTC, "CoreAudioCaptureDevice::getDeviceInfo failed to get device name with error %d (%.4s)", (int)err, (char*)&err);
        return false;
    }

    label = localizedName;
    CFRelease(localizedName);

    return true;
}

std::optional<CoreAudioCaptureDevice> CoreAudioCaptureDevice::create(uint32_t deviceID, DeviceType type, const String& groupID)
{
    ASSERT(type == CaptureDevice::DeviceType::Microphone || type == CaptureDevice::DeviceType::Speaker);
    String persistentID;
    String label;
    if (!getDeviceInfo(deviceID, type, persistentID, label))
        return std::nullopt;

    return CoreAudioCaptureDevice(deviceID, persistentID, type, label, groupID.isNull() ? persistentID : groupID);
}

CoreAudioCaptureDevice::CoreAudioCaptureDevice(uint32_t deviceID, const String& persistentID, DeviceType deviceType, const String& label, const String& groupID)
    : CaptureDevice(persistentID, deviceType, label, groupID)
    , m_deviceID(deviceID)
{
    AudioObjectPropertyAddress address {
        kAudioDevicePropertyDeviceIsAlive,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    UInt32 state = 0;
    UInt32 dataSize = sizeof(state);
    auto err = AudioObjectGetPropertyData(static_cast<UInt32>(m_deviceID), &address, 0, nullptr, &dataSize, &state);
    if (err)
        RELEASE_LOG_ERROR(WebRTC, "CoreAudioCaptureDevice::CoreAudioCaptureDevice(%p) failed to get \"is alive\" with error %d (%.4s)", this, (int)err, (char*)&err);
    setEnabled(!err && state);

    UInt32 property = deviceType == CaptureDevice::DeviceType::Microphone ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice;
    address = {
        property,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioDeviceID defaultID = kAudioDeviceUnknown;
    dataSize = sizeof(defaultID);
    err = AudioObjectGetPropertyData(kAudioObjectSystemObject, &address, 0, nullptr, &dataSize, &defaultID);
    if (err)
        RELEASE_LOG_ERROR(WebRTC, "CoreAudioCaptureDevice::CoreAudioCaptureDevice(%p) failed to get \"is default\" with error %d (%.4s)", this, (int)err, (char*)&err);
    setIsDefault(!err && defaultID == m_deviceID);
}

Vector<AudioDeviceID> CoreAudioCaptureDevice::relatedAudioDeviceIDs(AudioDeviceID deviceID)
{
    UInt32 size = 0;
    AudioObjectPropertyAddress property = {
        kAudioDevicePropertyRelatedDevices,
        kAudioDevicePropertyScopeOutput,
        kAudioObjectPropertyElementMain
    };
    OSStatus error = AudioObjectGetPropertyDataSize(deviceID, &property, 0, 0, &size);
    if (error || !size)
        return { };

    Vector<AudioDeviceID> devices(size / sizeof(AudioDeviceID));
    error = AudioObjectGetPropertyData(deviceID, &property, 0, nullptr, &size, devices.data());
    if (error)
        return { };
    return devices;
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && PLATFORM(MAC)
