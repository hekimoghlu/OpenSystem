/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "AudioDestinationCocoa.h"

#if ENABLE(WEB_AUDIO)

#include "AudioBus.h"
#include "AudioSession.h"
#include "AudioUtilities.h"
#include "Logging.h"
#include "MultiChannelResampler.h"
#include "PushPullFIFO.h"
#include "SharedAudioDestination.h"
#include "SpanCoreAudio.h"
#include <algorithm>

namespace WebCore {

constexpr size_t fifoSize = 96 * AudioUtilities::renderQuantumSize;

CreateAudioDestinationCocoaOverride AudioDestinationCocoa::createOverride = nullptr;

Ref<AudioDestination> AudioDestination::create(AudioIOCallback& callback, const String&, unsigned numberOfInputChannels, unsigned numberOfOutputChannels, float sampleRate)
{
    // FIXME: make use of inputDeviceId as appropriate.

    // FIXME: Add support for local/live audio input.
    if (numberOfInputChannels)
        WTFLogAlways("AudioDestination::create(%u, %u, %f) - unhandled input channels", numberOfInputChannels, numberOfOutputChannels, sampleRate);

    if (numberOfOutputChannels > AudioSession::sharedSession().maximumNumberOfOutputChannels())
        WTFLogAlways("AudioDestination::create(%u, %u, %f) - unhandled output channels", numberOfInputChannels, numberOfOutputChannels, sampleRate);

    if (AudioDestinationCocoa::createOverride)
        return AudioDestinationCocoa::createOverride(callback, sampleRate);

    return SharedAudioDestination::create(callback, numberOfOutputChannels, sampleRate, [numberOfOutputChannels, sampleRate] (AudioIOCallback& callback) {
        return adoptRef(*new AudioDestinationCocoa(callback, numberOfOutputChannels, sampleRate));
    });
}

float AudioDestination::hardwareSampleRate()
{
    return AudioSession::sharedSession().sampleRate();
}

unsigned long AudioDestination::maxChannelCount()
{
    return AudioSession::sharedSession().maximumNumberOfOutputChannels();
}

AudioDestinationCocoa::AudioDestinationCocoa(AudioIOCallback& callback, unsigned numberOfOutputChannels, float sampleRate)
    : AudioDestinationResampler(callback, numberOfOutputChannels, sampleRate, hardwareSampleRate())
    , m_audioOutputUnitAdaptor(*this)
{
    m_audioOutputUnitAdaptor.configure(hardwareSampleRate(), numberOfOutputChannels);
}

AudioDestinationCocoa::~AudioDestinationCocoa() = default;

void AudioDestinationCocoa::startRendering(CompletionHandler<void(bool)>&& completionHandler)
{
    ASSERT(isMainThread());
    auto success = m_audioOutputUnitAdaptor.start() == noErr;
    if (success)
        setIsPlaying(true);

    callOnMainThread([completionHandler = WTFMove(completionHandler), success]() mutable {
        completionHandler(success);
    });
}

void AudioDestinationCocoa::stopRendering(CompletionHandler<void(bool)>&& completionHandler)
{
    ASSERT(isMainThread());
    auto success = m_audioOutputUnitAdaptor.stop() == noErr;
    if (success)
        setIsPlaying(false);

    callOnMainThread([completionHandler = WTFMove(completionHandler), success]() mutable {
        completionHandler(success);
    });
}

// Pulls on our provider to get rendered audio stream.
OSStatus AudioDestinationCocoa::render(double sampleTime, uint64_t hostTime, UInt32 numberOfFrames, AudioBufferList* ioData)
{
    ASSERT(!isMainThread());

    auto numberOfBuffers = std::min<UInt32>(ioData->mNumberBuffers, m_outputBus->numberOfChannels());
    auto buffers = span(*ioData);

    // Associate the destination data array with the output bus then fill the FIFO.
    for (UInt32 i = 0; i < numberOfBuffers; ++i) {
        auto memory = mutableSpan<float>(buffers[i]);
        if (numberOfFrames < memory.size())
            memory = memory.first(numberOfFrames);
        m_outputBus->setChannelMemory(i, memory);
    }
    auto framesToRender = pullRendered(numberOfFrames);
    bool success = AudioDestinationResampler::render(sampleTime, MonotonicTime::fromMachAbsoluteTime(hostTime), framesToRender);
    return success ? noErr : -1;
}

MediaTime AudioDestinationCocoa::outputLatency() const
{
    return MediaTime { static_cast<int64_t>(m_audioOutputUnitAdaptor.outputLatency()), static_cast<uint32_t>(sampleRate()) } + MediaTime { static_cast<int64_t>(AudioSession::protectedSharedSession()->outputLatency()), static_cast<uint32_t>(AudioSession::protectedSharedSession()->sampleRate()) };
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
