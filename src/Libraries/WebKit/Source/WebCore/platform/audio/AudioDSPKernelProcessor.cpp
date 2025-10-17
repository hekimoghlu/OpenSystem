/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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

#include "AudioDSPKernelProcessor.h"

#include "AudioDSPKernel.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioDSPKernelProcessor);

// setNumberOfChannels() may later be called if the object is not yet in an "initialized" state.
AudioDSPKernelProcessor::AudioDSPKernelProcessor(float sampleRate, unsigned numberOfChannels)
    : AudioProcessor(sampleRate, numberOfChannels)
{
}

AudioDSPKernelProcessor::~AudioDSPKernelProcessor() = default;

void AudioDSPKernelProcessor::initialize()
{
    if (isInitialized())
        return;

    ASSERT(!m_kernels.size());
    {
        // Heap allocations are forbidden on the audio thread for performance reasons so we need to
        // explicitly allow the following allocation(s).
        DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;

        // Create processing kernels, one per channel.
        m_kernels = Vector<std::unique_ptr<AudioDSPKernel>>(numberOfChannels(), [this](size_t) {
            return createKernel();
        });
    }
    m_initialized = true;
    m_hasJustReset = true;
}

void AudioDSPKernelProcessor::uninitialize()
{
    if (!isInitialized())
        return;
        
    m_kernels.clear();

    m_initialized = false;
}

void AudioDSPKernelProcessor::process(const AudioBus* source, AudioBus* destination, size_t framesToProcess)
{
    ASSERT(source && destination);
    if (!source || !destination)
        return;
        
    if (!isInitialized()) {
        destination->zero();
        return;
    }

    bool channelCountMatches = source->numberOfChannels() == destination->numberOfChannels() && source->numberOfChannels() == m_kernels.size();
    ASSERT(channelCountMatches);
    if (!channelCountMatches)
        return;
        
    for (unsigned i = 0; i < m_kernels.size(); ++i)
        m_kernels[i]->process(source->channel(i)->span().first(framesToProcess), destination->channel(i)->mutableSpan());
}

void AudioDSPKernelProcessor::processOnlyAudioParams(size_t framesToProcess)
{
    if (!isInitialized())
        return;

    for (unsigned i = 0; i < m_kernels.size(); ++i)
        m_kernels[i]->processOnlyAudioParams(framesToProcess);
}

// Resets filter state
void AudioDSPKernelProcessor::reset()
{
    if (!isInitialized())
        return;

    // Forces snap to parameter values - first time.
    // Any processing depending on this value must set it to false at the appropriate time.
    m_hasJustReset = true;

    for (unsigned i = 0; i < m_kernels.size(); ++i)
        m_kernels[i]->reset();
}

void AudioDSPKernelProcessor::setNumberOfChannels(unsigned numberOfChannels)
{
    if (numberOfChannels == m_numberOfChannels)
        return;
        
    ASSERT(!isInitialized());
    if (!isInitialized())
        m_numberOfChannels = numberOfChannels;
}

double AudioDSPKernelProcessor::tailTime() const
{
    // It is expected that all the kernels have the same tailTime.
    return !m_kernels.isEmpty() ? m_kernels.first()->tailTime() : 0;
}

double AudioDSPKernelProcessor::latencyTime() const
{
    // It is expected that all the kernels have the same latencyTime.
    return !m_kernels.isEmpty() ? m_kernels.first()->latencyTime() : 0;
}

bool AudioDSPKernelProcessor::requiresTailProcessing() const
{
    // Always return true even if the tail time and latency might both be zero.
    return true;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
