/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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

#if ENABLE(WEB_AUDIO)

#include <pal/cf/AudioToolboxSoftLink.h>

namespace WebCore {

AudioOutputUnitAdaptor::AudioOutputUnitAdaptor(AudioUnitRenderer& renderer)
    : m_outputUnit(0)
    , m_audioUnitRenderer(renderer)
{
}

AudioOutputUnitAdaptor::~AudioOutputUnitAdaptor()
{
    if (m_outputUnit)
        PAL::AudioComponentInstanceDispose(m_outputUnit);
}

OSStatus AudioOutputUnitAdaptor::start()
{
    auto result = PAL::AudioOutputUnitStart(m_outputUnit);
    if (result != noErr)
        WTFLogAlways("ERROR: AudioOutputUnitStart() call failed with error code: %ld", static_cast<long>(result));
    return result;
}

OSStatus AudioOutputUnitAdaptor::stop()
{
    return PAL::AudioOutputUnitStop(m_outputUnit);
}

// DefaultOutputUnit callback
OSStatus AudioOutputUnitAdaptor::inputProc(void* userData, AudioUnitRenderActionFlags*, const AudioTimeStamp* timeStamp, UInt32 /*busNumber*/, UInt32 numberOfFrames, AudioBufferList* ioData)
{
    auto* adaptor = static_cast<AudioOutputUnitAdaptor*>(userData);
    double sampleTime = 0.;
    uint64_t hostTime = 0;
    if (timeStamp) {
        sampleTime = timeStamp->mSampleTime;
        hostTime = timeStamp->mHostTime;
    }

    return adaptor->m_audioUnitRenderer.render(sampleTime, hostTime, numberOfFrames, ioData);
}

size_t AudioOutputUnitAdaptor::outputLatency() const
{
    Float64 latency = 0;
    UInt32 size = sizeof(latency);
    if (PAL::AudioUnitGetProperty(m_outputUnit, kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0, &latency, &size) == noErr)
        return latency;
    return 0;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
