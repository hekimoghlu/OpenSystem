/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#include "AudioDestinationResampler.h"

#if ENABLE(WEB_AUDIO)

#include "AudioBus.h"
#include "AudioSession.h"
#include "AudioUtilities.h"
#include "Logging.h"
#include "MultiChannelResampler.h"

namespace WebCore {

constexpr size_t fifoSize = 96 * AudioUtilities::renderQuantumSize;

AudioDestinationResampler::AudioDestinationResampler(AudioIOCallback& callback, unsigned numberOfOutputChannels, float inputSampleRate, float outputSampleRate)
    : AudioDestination(callback, inputSampleRate)
    , m_outputBus(AudioBus::create(numberOfOutputChannels, AudioUtilities::renderQuantumSize, false).releaseNonNull())
    , m_renderBus(AudioBus::create(numberOfOutputChannels, AudioUtilities::renderQuantumSize).releaseNonNull())
    , m_fifo(numberOfOutputChannels, fifoSize)
{
    if (inputSampleRate != outputSampleRate) {
        double scaleFactor = static_cast<double>(inputSampleRate) / outputSampleRate;
        m_resampler = makeUnique<MultiChannelResampler>(scaleFactor, numberOfOutputChannels, AudioUtilities::renderQuantumSize, [this](AudioBus* bus, size_t framesToProcess) {
            ASSERT_UNUSED(framesToProcess, framesToProcess == AudioUtilities::renderQuantumSize);
            callRenderCallback(nullptr, bus, AudioUtilities::renderQuantumSize, m_outputTimestamp);
        });
    }
}

AudioDestinationResampler::~AudioDestinationResampler() = default;

unsigned AudioDestinationResampler::framesPerBuffer() const
{
    return m_renderBus->length();
}

void AudioDestinationResampler::start(Function<void(Function<void()>&&)>&& dispatchToRenderThread, CompletionHandler<void(bool)>&& completionHandler)
{
    ASSERT(isMainThread());
    LOG(Media, "AudioDestinationResampler::start");
    {
        Locker locker { m_dispatchToRenderThreadLock };
        m_dispatchToRenderThread = WTFMove(dispatchToRenderThread);
    }
    startRendering(WTFMove(completionHandler));
}

void AudioDestinationResampler::stop(CompletionHandler<void(bool)>&& completionHandler)
{
    ASSERT(isMainThread());
    LOG(Media, "AudioDestinationResampler::stop");
    stopRendering(WTFMove(completionHandler));
    {
        Locker locker { m_dispatchToRenderThreadLock };
        m_dispatchToRenderThread = nullptr;
    }
}

void AudioDestinationResampler::setIsPlaying(bool isPlaying)
{
    ASSERT(isMainThread());

    if (m_isPlaying == isPlaying)
        return;

    m_isPlaying = isPlaying;

    {
        Locker locker { m_callbackLock };
        if (m_callback)
            m_callback->isPlayingDidChange();
    }
}

size_t AudioDestinationResampler::pullRendered(size_t numberOfFrames)
{
    ASSERT(!isMainThread());
    numberOfFrames = std::min(numberOfFrames, fifoSize);
    Locker locker { m_fifoLock };
    return m_fifo.pull(m_outputBus.ptr(), numberOfFrames);
}

bool AudioDestinationResampler::render(double sampleTime, MonotonicTime hostTime,  size_t framesToRender)
{
    m_outputTimestamp = {
        Seconds { sampleTime },
        hostTime
    };
    // When there is a AudioWorklet, we do rendering on the AudioWorkletThread.
    if (!m_dispatchToRenderThreadLock.tryLock())
        return false;

    Locker locker { AdoptLock, m_dispatchToRenderThreadLock };
    if (!m_dispatchToRenderThread)
        renderOnRenderingThreadIfPlaying(framesToRender);
    else {
        m_dispatchToRenderThread([protectedThis = Ref { *this }, framesToRender]() mutable {
            protectedThis->renderOnRenderingThreadIfPlaying(framesToRender);
        });
    }
    return true;
}

// This runs on the AudioWorkletThread when AudioWorklet is enabled, on the audio device's rendering thread otherwise.
void AudioDestinationResampler::renderOnRenderingThreadIfPlaying(size_t framesToRender)
{
    if (!m_isPlaying)
        return;
    for (size_t pushedFrames = 0; pushedFrames < framesToRender; pushedFrames += AudioUtilities::renderQuantumSize) {
        if (m_resampler)
            m_resampler->process(m_renderBus.ptr(), AudioUtilities::renderQuantumSize);
        else
            callRenderCallback(nullptr, m_renderBus.ptr(), AudioUtilities::renderQuantumSize, m_outputTimestamp);

        Locker locker { m_fifoLock };
        m_fifo.push(m_renderBus.ptr());
    }
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
