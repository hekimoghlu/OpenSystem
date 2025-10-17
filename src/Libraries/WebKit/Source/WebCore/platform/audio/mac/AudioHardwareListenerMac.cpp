/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#include "AudioHardwareListenerMac.h"

#if PLATFORM(MAC)

#include <algorithm>
#include <wtf/StdLibExtras.h>

enum {
    kAudioHardwarePropertyProcessIsRunning = 'prun'
};

namespace WebCore {
    
static AudioHardwareActivityType isAudioHardwareProcessRunning()
{
    AudioObjectPropertyAddress propertyAddress = {
        kAudioHardwarePropertyProcessIsRunning,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    
    if (!AudioObjectHasProperty(kAudioObjectSystemObject, &propertyAddress))
        return AudioHardwareActivityType::Unknown;
    
    UInt32 result = 0;
    UInt32 resultSize = sizeof(UInt32);

    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, 0, &resultSize, &result))
        return AudioHardwareActivityType::Unknown;

    if (result)
        return AudioHardwareActivityType::IsActive;
    else
        return AudioHardwareActivityType::IsInactive;
}

static AudioHardwareListener::BufferSizeRange currentDeviceSupportedBufferSizes()
{
    AudioDeviceID deviceID = kAudioDeviceUnknown;
    UInt32 descriptorSize = sizeof(deviceID);
    AudioObjectPropertyAddress defaultOutputDeviceDescriptor = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &defaultOutputDeviceDescriptor, 0, 0, &descriptorSize, (void*)&deviceID))
        return { };

    AudioValueRange bufferSizes;
    descriptorSize = sizeof(bufferSizes);

    AudioObjectPropertyAddress bufferSizeDescriptor = {
        kAudioDevicePropertyBufferFrameSizeRange,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    if (AudioObjectGetPropertyData(deviceID, &bufferSizeDescriptor, 0, 0, &descriptorSize, &bufferSizes))
        return { };

    return { static_cast<size_t>(bufferSizes.mMinimum), static_cast<size_t>(bufferSizes.mMaximum) };
}


static const AudioObjectPropertyAddress& processIsRunningPropertyDescriptor()
{
    static const AudioObjectPropertyAddress processIsRunningProperty = {
        kAudioHardwarePropertyProcessIsRunning,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    return processIsRunningProperty;
}

static const AudioObjectPropertyAddress& outputDevicePropertyDescriptor()
{
    static const AudioObjectPropertyAddress outputDeviceProperty = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    return outputDeviceProperty;
}

Ref<AudioHardwareListenerMac> AudioHardwareListenerMac::create(Client& client)
{
    return adoptRef(*new AudioHardwareListenerMac(client));
}

AudioHardwareListenerMac::AudioHardwareListenerMac(Client& client)
    : AudioHardwareListener(client)
{
    setHardwareActivity(isAudioHardwareProcessRunning());
    setSupportedBufferSizes(currentDeviceSupportedBufferSizes());

    WeakPtr weakThis { *this };
    m_block = Block_copy(^(UInt32 count, const AudioObjectPropertyAddress properties[]) {
        if (weakThis)
            weakThis->propertyChanged(unsafeMakeSpan(properties, count));
    });

    AudioObjectAddPropertyListenerBlock(kAudioObjectSystemObject, &processIsRunningPropertyDescriptor(), dispatch_get_main_queue(), m_block);
    AudioObjectAddPropertyListenerBlock(kAudioObjectSystemObject, &outputDevicePropertyDescriptor(), dispatch_get_main_queue(), m_block);
}

AudioHardwareListenerMac::~AudioHardwareListenerMac()
{
    AudioObjectRemovePropertyListenerBlock(kAudioObjectSystemObject, &processIsRunningPropertyDescriptor(), dispatch_get_main_queue(), m_block);
    AudioObjectRemovePropertyListenerBlock(kAudioObjectSystemObject, &outputDevicePropertyDescriptor(), dispatch_get_main_queue(), m_block);
    Block_release(m_block);
}

void AudioHardwareListenerMac::propertyChanged(std::span<const AudioObjectPropertyAddress> properties)
{
    auto deviceRunning = asByteSpan(processIsRunningPropertyDescriptor());
    auto outputDevice = asByteSpan(outputDevicePropertyDescriptor());

    for (auto& property : properties) {
        auto propertyBytes = asByteSpan(property);
        if (equalSpans(propertyBytes, deviceRunning))
            processIsRunningChanged();
        else if (equalSpans(propertyBytes, outputDevice))
            outputDeviceChanged();
    }
}

void AudioHardwareListenerMac::processIsRunningChanged()
{
    AudioHardwareActivityType activity = isAudioHardwareProcessRunning();
    if (activity == hardwareActivity())
        return;
    setHardwareActivity(activity);
    
    if (hardwareActivity() == AudioHardwareActivityType::IsActive)
        m_client.audioHardwareDidBecomeActive();
    else if (hardwareActivity() == AudioHardwareActivityType::IsInactive)
        m_client.audioHardwareDidBecomeInactive();
}

void AudioHardwareListenerMac::outputDeviceChanged()
{
    setSupportedBufferSizes(currentDeviceSupportedBufferSizes());
    m_client.audioOutputDeviceChanged();
}

}

#endif
