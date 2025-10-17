/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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

#include "MultiChannelResampler.h"

#include "AudioBus.h"
#include "SincResampler.h"
#include <functional>
#include <wtf/Algorithms.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MultiChannelResampler);

MultiChannelResampler::MultiChannelResampler(double scaleFactor, unsigned numberOfChannels, unsigned requestFrames, Function<void(AudioBus*, size_t framesToProcess)>&& provideInput)
    : m_numberOfChannels(numberOfChannels)
    , m_provideInput(WTFMove(provideInput))
    , m_multiChannelBus(AudioBus::create(numberOfChannels, requestFrames))
{
    // Create each channel's resampler.
    m_kernels = Vector<std::unique_ptr<SincResampler>>(numberOfChannels, [&](size_t channelIndex) {
        return makeUnique<SincResampler>(scaleFactor, requestFrames, std::bind(&MultiChannelResampler::provideInputForChannel, this, std::placeholders::_1, std::placeholders::_2, channelIndex));
    });
}

MultiChannelResampler::~MultiChannelResampler() = default;

void MultiChannelResampler::process(AudioBus* destination, size_t framesToProcess)
{
    ASSERT(m_numberOfChannels == destination->numberOfChannels());
    if (destination->numberOfChannels() == 1) {
        // Fast path when the bus is mono to avoid the chunking below.
        m_kernels[0]->process(destination->channel(0)->mutableSpan(), framesToProcess);
        return;
    }

    // We need to ensure that SincResampler only calls provideInputForChannel() once for each channel or it will confuse the logic
    // inside provideInputForChannel(). To ensure this, we chunk the number of requested frames into SincResampler::chunkSize()
    // sized chunks. SincResampler guarantees it will only call provideInputForChannel() once once we resample this way.
    m_outputFramesReady = 0;
    while (m_outputFramesReady < framesToProcess) {
        size_t chunkSize = m_kernels[0]->chunkSize();
        size_t framesThisTime = std::min(framesToProcess - m_outputFramesReady, chunkSize);

        for (unsigned channelIndex = 0; channelIndex < m_numberOfChannels; ++channelIndex) {
            ASSERT(chunkSize == m_kernels[channelIndex]->chunkSize());
            auto* channel = destination->channel(channelIndex);
            m_kernels[channelIndex]->process(channel->mutableSpan().subspan(m_outputFramesReady), framesThisTime);
        }

        m_outputFramesReady += framesThisTime;
    }
}

void MultiChannelResampler::provideInputForChannel(std::span<float> buffer, size_t framesToProcess, unsigned channelIndex)
{
    ASSERT(channelIndex < m_multiChannelBus->numberOfChannels());
    ASSERT(framesToProcess <= m_multiChannelBus->length());

    if (!channelIndex)
        m_provideInput(m_multiChannelBus.get(), framesToProcess);

    // Copy the channel data from what we received from m_multiChannelProvider.
    memcpySpan(buffer, m_multiChannelBus->channel(channelIndex)->span().first(framesToProcess));
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
