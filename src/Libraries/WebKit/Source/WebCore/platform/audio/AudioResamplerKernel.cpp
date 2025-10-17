/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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

#if ENABLE(WEB_AUDIO)

#include "AudioResamplerKernel.h"

#include "AudioResampler.h"
#include "AudioUtilities.h"
#include <algorithm>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioResamplerKernel);

AudioResamplerKernel::AudioResamplerKernel(AudioResampler* resampler)
    : m_resampler(resampler)
    // The buffer size must be large enough to hold up to two extra sample frames for the linear interpolation.
    , m_sourceBuffer(2 + static_cast<int>(AudioUtilities::renderQuantumSize * AudioResampler::MaxRate))
{
    m_lastValues[0] = 0.0f;
    m_lastValues[1] = 0.0f;
}

std::span<float> AudioResamplerKernel::getSourceSpan(size_t framesToProcess, size_t* numberOfSourceFramesNeededP)
{
    ASSERT(framesToProcess <= AudioUtilities::renderQuantumSize);
    
    // Calculate the next "virtual" index.  After process() is called, m_virtualReadIndex will equal this value.
    double nextFractionalIndex = m_virtualReadIndex + framesToProcess * rate();

    // Because we're linearly interpolating between the previous and next sample we need to round up so we include the next sample.
    int endIndex = static_cast<int>(nextFractionalIndex + 1.0); // round up to next integer index

    // Determine how many input frames we'll need.
    // We need to fill the buffer up to and including endIndex (so add 1) but we've already buffered m_fillIndex frames from last time.
    size_t framesNeeded = 1 + endIndex - m_fillIndex;
    if (numberOfSourceFramesNeededP)
        *numberOfSourceFramesNeededP = framesNeeded;

    // Do bounds checking for the source buffer.
    bool isGood = m_fillIndex < m_sourceBuffer.size() && m_fillIndex + framesNeeded <= m_sourceBuffer.size();
    ASSERT(isGood);
    if (!isGood)
        return { };

    return m_sourceBuffer.span().subspan(m_fillIndex);
}

void AudioResamplerKernel::process(std::span<float> destination, size_t framesToProcess)
{
    ASSERT(framesToProcess <= AudioUtilities::renderQuantumSize);

    auto source = m_sourceBuffer.span();
    
    double rate = this->rate();
    rate = std::max(0.0, rate);
    rate = std::min(AudioResampler::MaxRate, rate);
    
    // Start out with the previous saved values (if any).
    if (m_fillIndex > 0) {
        source[0] = m_lastValues[0];
        source[1] = m_lastValues[1];
    }

    // Make a local copy.
    double virtualReadIndex = m_virtualReadIndex;
    
    // Sanity check source buffer access.
    ASSERT(framesToProcess > 0);
    ASSERT(virtualReadIndex >= 0 && 1 + static_cast<unsigned>(virtualReadIndex + (framesToProcess - 1) * rate) < m_sourceBuffer.size());

    // Do the linear interpolation.
    int n = framesToProcess;
    size_t destinationIndex = 0;
    while (n--) {
        unsigned readIndex = static_cast<unsigned>(virtualReadIndex);
        double interpolationFactor = virtualReadIndex - readIndex;

        double sample1 = source[readIndex];
        double sample2 = source[readIndex + 1];

        double sample = (1.0 - interpolationFactor) * sample1 + interpolationFactor * sample2;

        destination[destinationIndex++] = static_cast<float>(sample);

        virtualReadIndex += rate;
    }

    // Save the last two sample-frames which will later be used at the beginning of the source buffer the next time around.
    int readIndex = static_cast<int>(virtualReadIndex);
    m_lastValues[0] = source[readIndex];
    m_lastValues[1] = source[readIndex + 1];
    m_fillIndex = 2;

    // Wrap the virtual read index back to the start of the buffer.
    virtualReadIndex -= readIndex;

    // Put local copy back into member variable.
    m_virtualReadIndex = virtualReadIndex;
}

void AudioResamplerKernel::reset()
{
    m_virtualReadIndex = 0.0;
    m_fillIndex = 0;
    m_lastValues[0] = 0.0f;
    m_lastValues[1] = 0.0f;
}

double AudioResamplerKernel::rate() const
{
    return m_resampler->rate();
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
