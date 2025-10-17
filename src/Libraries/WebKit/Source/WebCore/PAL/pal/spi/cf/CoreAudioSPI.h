/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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
#pragma once

#if PLATFORM(COCOA)

#include <CoreAudio/CoreAudioTypes.h>

#if USE(APPLE_INTERNAL_SDK)
#include <CoreAudio/AudioHardwarePriv.h>
#else

#if PLATFORM(MAC)
#include <CoreAudio/AudioHardware.h>

CF_ENUM(AudioObjectPropertySelector)
{
#if HAVE(AUDIO_DEVICE_PROPERTY_REFERENCE_STREAM_ENABLED)
    kAudioDevicePropertyReferenceStreamEnabled = 'tapd',
#else
    kAudioDevicePropertyTapEnabled = 'tapd',
#endif
};

#else

WTF_EXTERN_C_BEGIN

typedef UInt32 AudioObjectPropertySelector;
typedef UInt32 AudioObjectPropertyScope;
typedef UInt32 AudioObjectPropertyElement;

struct AudioObjectPropertyAddress {
    AudioObjectPropertySelector mSelector;
    AudioObjectPropertyScope mScope;
    AudioObjectPropertyElement mElement;
};
typedef struct AudioObjectPropertyAddress AudioObjectPropertyAddress;

CF_ENUM(AudioObjectPropertyScope)
{
    kAudioObjectPropertyScopeGlobal = 'glob'
};

CF_ENUM(AudioObjectPropertySelector)
{
    kAudioHardwarePropertyDefaultInputDevice = 'dIn ',
};

CF_ENUM(int)
{
    kAudioObjectSystemObject    = 1
};

typedef UInt32  AudioObjectID;
typedef AudioObjectID AudioDeviceID;

extern Boolean AudioObjectHasProperty(AudioObjectID inObjectID, const AudioObjectPropertyAddress* __nullable inAddress);
extern OSStatus AudioObjectGetPropertyData(AudioObjectID inObjectID, const AudioObjectPropertyAddress* __nullable inAddress, UInt32                              inQualifierDataSize, const void* __nullable inQualifierData, UInt32* __nullable ioDataSize, void* __nullable outData);

WTF_EXTERN_C_END

#endif

#endif

WTF_EXTERN_C_BEGIN

extern OSStatus AudioDeviceDuck(AudioDeviceID inDevice, Float32 inDuckedLevel, const AudioTimeStamp* __nullable inStartTime, Float32 inRampDuration);

WTF_EXTERN_C_END

#endif // PLATFORM(COCOA)
