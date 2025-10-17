/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
#import "AVAudioSessionCaptureDevice.h"

#if ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)

#import <AVFoundation/AVAudioSession.h>

#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

AVAudioSessionCaptureDevice AVAudioSessionCaptureDevice::createInput(AVAudioSessionPortDescription* deviceInput, AVAudioSessionPortDescription *defaultInput)
{
    return AVAudioSessionCaptureDevice(deviceInput, defaultInput, CaptureDevice::DeviceType::Microphone);
}

AVAudioSessionCaptureDevice AVAudioSessionCaptureDevice::createOutput(AVAudioSessionPortDescription *deviceOutput, AVAudioSessionPortDescription *defaultOutput)
{
    return AVAudioSessionCaptureDevice(deviceOutput, defaultOutput, CaptureDevice::DeviceType::Speaker);
}

AVAudioSessionCaptureDevice::AVAudioSessionCaptureDevice(AVAudioSessionPortDescription *device, AVAudioSessionPortDescription *defaultDevice, CaptureDevice::DeviceType deviceType)
    : CaptureDevice(device.UID, deviceType, device.portName)
    , m_description(device)
{
    setEnabled(true);
    setIsDefault(defaultDevice && [defaultDevice.UID isEqualToString:device.UID]);
}

bool AVAudioSessionCaptureDevice::isBuiltin() const
{
    if (type() == CaptureDevice::DeviceType::Microphone)
        return [m_description portType] == AVAudioSessionPortBuiltInMic;

    return [m_description portType] == AVAudioSessionPortBuiltInReceiver || [m_description portType] == AVAudioSessionPortBuiltInSpeaker;
}

bool AVAudioSessionCaptureDevice::isLineInOrOut() const
{
    return [m_description portType] == (type() == CaptureDevice::DeviceType::Microphone ? AVAudioSessionPortLineIn : AVAudioSessionPortLineOut);
}

bool AVAudioSessionCaptureDevice::isHeadset() const
{
    return [m_description portType] == (type() == CaptureDevice::DeviceType::Microphone ? AVAudioSessionPortHeadsetMic : AVAudioSessionPortHeadphones);
}

AVAudioSessionCaptureDevice::AVAudioSessionCaptureDevice(const String& persistentId, DeviceType type, const String& label, const String& groupId, bool isEnabled, bool isDefault, bool isMock, RetainPtr<AVAudioSessionPortDescription>&& description)
    : CaptureDevice(persistentId, type, label, groupId, isEnabled, isDefault, isMock)
    , m_description(WTFMove(description))
{
}

AVAudioSessionCaptureDevice AVAudioSessionCaptureDevice::isolatedCopy() &&
{
    return {
        WTFMove(m_persistentId).isolatedCopy(),
        m_type,
        WTFMove(m_label).isolatedCopy(),
        WTFMove(m_groupId).isolatedCopy(),
        m_enabled,
        m_default,
        m_isMockDevice,
        WTFMove(m_description)
    };
}

}

#endif // ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)
