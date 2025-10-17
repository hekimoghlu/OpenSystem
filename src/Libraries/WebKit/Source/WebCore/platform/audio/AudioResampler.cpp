/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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

#include "AudioResampler.h"

#include "AudioBus.h"
#include <algorithm>
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioResampler);

AudioResampler::AudioResampler()
{
    m_kernels.append(makeUnique<AudioResamplerKernel>(this));
    m_sourceBus = AudioBus::create(1, 0, false);
}

AudioResampler::AudioResampler(unsigned numberOfChannels)
{
    m_kernels = Vector<std::unique_ptr<AudioResamplerKernel>>(numberOfChannels, [&](size_t) {
        return makeUnique<AudioResamplerKernel>(this);
    });

    m_sourceBus = AudioBus::create(numberOfChannels, 0, false);
}

void AudioResampler::configureChannels(unsigned numberOfChannels)
{
    unsigned currentSize = m_kernels.size();
    if (numberOfChannels == currentSize)
        return; // already setup

    // First deal with adding or removing kernels.
    if (numberOfChannels > currentSize) {
        for (unsigned i = currentSize; i < numberOfChannels; ++i)
            m_kernels.append(makeUnique<AudioResamplerKernel>(this));
    } else
        m_kernels.resize(numberOfChannels);

    // Reconfigure our source bus to the new channel size.
    m_sourceBus = AudioBus::create(numberOfChannels, 0, false);
}

void AudioResampler::process(AudioSourceProvider* provider, AudioBus* destinationBus, size_t framesToProcess)
{
    ASSERT(provider);
    if (!provider)
        return;
        
    unsigned numberOfChannels = m_kernels.size();

    // Make sure our configuration matches the bus we're rendering to.
    bool channelsMatch = (destinationBus && destinationBus->numberOfChannels() == numberOfChannels);
    ASSERT(channelsMatch);
    if (!channelsMatch)
        return;

    // Setup the source bus.
    for (unsigned i = 0; i < numberOfChannels; ++i) {
        // Figure out how many frames we need to get from the provider, and a pointer to the buffer.
        size_t framesNeeded;
        std::span<float> fillSpan = m_kernels[i]->getSourceSpan(framesToProcess, &framesNeeded);
        ASSERT(!fillSpan.empty());
        if (fillSpan.empty())
            return;

        m_sourceBus->setChannelMemory(i, fillSpan.first(framesNeeded));
    }

    // Ask the provider to supply the desired number of source frames.
    provider->provideInput(m_sourceBus.get(), m_sourceBus->length());

    // Now that we have the source data, resample each channel into the destination bus.
    // FIXME: optimize for the common stereo case where it's faster to process both left/right channels in the same inner loop.
    for (unsigned i = 0; i < numberOfChannels; ++i) {
        auto destination = destinationBus->channel(i)->mutableSpan();
        m_kernels[i]->process(destination, framesToProcess);
    }
}

void AudioResampler::setRate(double rate)
{
    if (std::isnan(rate) || std::isinf(rate) || rate <= 0.0)
        return;
    
    m_rate = std::min(AudioResampler::MaxRate, rate);
}

void AudioResampler::reset()
{
    unsigned numberOfChannels = m_kernels.size();
    for (unsigned i = 0; i < numberOfChannels; ++i)
        m_kernels[i]->reset();
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
