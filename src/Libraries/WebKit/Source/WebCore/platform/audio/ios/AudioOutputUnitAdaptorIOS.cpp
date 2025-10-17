/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#include "AudioOutputUnitAdaptor.h"

#if ENABLE(WEB_AUDIO) && PLATFORM(IOS_FAMILY)

#include "AudioSession.h"
#include <pal/cf/AudioToolboxSoftLink.h>

namespace WebCore {

void AudioOutputUnitAdaptor::configure(float hardwareSampleRate, unsigned numberOfOutputChannels)
{
    const int kPreferredBufferSize = 256;

    // Open and initialize DefaultOutputUnit
    AudioComponent comp;
    AudioComponentDescription desc;

    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = kAudioUnitSubType_RemoteIO;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    desc.componentFlags = 0;
    desc.componentFlagsMask = 0;
    comp = PAL::AudioComponentFindNext(0, &desc);

    ASSERT(comp);

    OSStatus result = PAL::AudioComponentInstanceNew(comp, &m_outputUnit);
    ASSERT_UNUSED(result, !result);

    UInt32 flag = 1;
    result = PAL::AudioUnitSetProperty(m_outputUnit,
        kAudioOutputUnitProperty_EnableIO,
        kAudioUnitScope_Output,
        0,
        &flag,
        sizeof(flag));
    ASSERT_UNUSED(result, !result);

    result = PAL::AudioUnitInitialize(m_outputUnit);
    ASSERT_UNUSED(result, !result);
    // Set render callback
    AURenderCallbackStruct input;
    input.inputProc = inputProc;
    input.inputProcRefCon = this;
    result = PAL::AudioUnitSetProperty(m_outputUnit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &input, sizeof(input));
    ASSERT_UNUSED(result, !result);

    // Set stream format
    AudioStreamBasicDescription streamFormat;

    UInt32 size = sizeof(AudioStreamBasicDescription);
    result = PAL::AudioUnitGetProperty(m_outputUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0, (void*)&streamFormat, &size);
    ASSERT_UNUSED(result, !result);

    constexpr int bytesPerFloat = sizeof(Float32);
    constexpr int bitsPerByte = 8;
    streamFormat.mSampleRate = hardwareSampleRate;
    streamFormat.mFormatID = kAudioFormatLinearPCM;
    streamFormat.mFormatFlags = static_cast<AudioFormatFlags>(kAudioFormatFlagsNativeFloatPacked) | static_cast<AudioFormatFlags>(kAudioFormatFlagIsNonInterleaved);
    streamFormat.mBytesPerPacket = bytesPerFloat;
    streamFormat.mFramesPerPacket = 1;
    streamFormat.mBytesPerFrame = bytesPerFloat;
    streamFormat.mChannelsPerFrame = numberOfOutputChannels;
    streamFormat.mBitsPerChannel = bitsPerByte * bytesPerFloat;

    result = PAL::AudioUnitSetProperty(m_outputUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, (void*)&streamFormat, sizeof(AudioStreamBasicDescription));
    ASSERT_UNUSED(result, !result);

    AudioSession::sharedSession().setPreferredBufferSize(kPreferredBufferSize);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO) && PLATFORM(IOS_FAMILY)

