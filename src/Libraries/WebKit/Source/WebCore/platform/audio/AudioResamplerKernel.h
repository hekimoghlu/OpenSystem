/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#ifndef AudioResamplerKernel_h
#define AudioResamplerKernel_h

#include "AudioArray.h"

#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AudioResampler;

// AudioResamplerKernel does resampling on a single mono channel.
// It uses a simple linear interpolation for good performance.

class AudioResamplerKernel final {
    WTF_MAKE_TZONE_ALLOCATED(AudioResamplerKernel);
    WTF_MAKE_NONCOPYABLE(AudioResamplerKernel);
public:
    AudioResamplerKernel(AudioResampler*);

    // getSourceSpan() should be called each time before process() is called.
    // Given a number of frames to process (for subsequent call to process()), it returns a span and numberOfSourceFramesNeeded
    // where sample data should be copied. This sample data provides the input to the resampler when process() is called.
    // framesToProcess must be less than or equal to AudioUtilities::renderQuantumSize.
    std::span<float> getSourceSpan(size_t framesToProcess, size_t* numberOfSourceFramesNeeded);

    // process() resamples framesToProcess frames from the source into destination.
    // Each call to process() must be preceded by a call to getSourceSpan() so that source input may be supplied.
    // framesToProcess must be less than or equal to AudioUtilities::renderQuantumSize.
    void process(std::span<float> destination, size_t framesToProcess);

    // Resets the processing state.
    void reset();

private:
    double rate() const;

    AudioResampler* m_resampler;
    AudioFloatArray m_sourceBuffer;
    
    // This is a (floating point) read index on the input stream.
    double m_virtualReadIndex { 0 };

    // We need to have continuity from one call of process() to the next.
    // m_lastValues stores the last two sample values from the last call to process().
    // m_fillIndex represents how many buffered samples we have which can be as many as 2.
    // For the first call to process() (or after reset()) there will be no buffered samples.
    std::array<float, 2> m_lastValues;
    unsigned m_fillIndex { 0 };
};

} // namespace WebCore

#endif // AudioResamplerKernel_h
