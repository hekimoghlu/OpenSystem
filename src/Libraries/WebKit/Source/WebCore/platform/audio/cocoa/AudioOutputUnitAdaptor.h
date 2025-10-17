/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

#if ENABLE(WEB_AUDIO)

#include <AudioUnit/AudioUnit.h>

namespace WebCore {

class AudioUnitRenderer {
public:
    virtual ~AudioUnitRenderer() = default;
    virtual OSStatus render(double sampleTime, uint64_t hostTime, UInt32 numberOfFrames, AudioBufferList* ioData) = 0;
};

class AudioOutputUnitAdaptor {
public:
    WEBCORE_EXPORT AudioOutputUnitAdaptor(AudioUnitRenderer&);
    WEBCORE_EXPORT ~AudioOutputUnitAdaptor();

    WEBCORE_EXPORT void configure(float hardwareSampleRate, unsigned numberOfOutputChannels);
    WEBCORE_EXPORT OSStatus start();
    WEBCORE_EXPORT OSStatus stop();

    WEBCORE_EXPORT size_t outputLatency() const;

private:
    static OSStatus inputProc(void* userData, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32 busNumber, UInt32 numberOfFrames, AudioBufferList* ioData);

    AudioUnit m_outputUnit;
    AudioUnitRenderer& m_audioUnitRenderer;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
