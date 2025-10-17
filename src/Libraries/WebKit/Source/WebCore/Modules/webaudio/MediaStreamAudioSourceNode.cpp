/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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

#if ENABLE(WEB_AUDIO) && ENABLE(MEDIA_STREAM)

#include "MediaStreamAudioSourceNode.h"

#include "AudioContext.h"
#include "AudioNodeOutput.h"
#include "AudioSourceProvider.h"
#include "AudioUtilities.h"
#include "Logging.h"
#include "MediaStreamAudioSourceOptions.h"
#include "WebAudioSourceProvider.h"
#include <wtf/Locker.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MediaStreamAudioSourceNode);

ExceptionOr<Ref<MediaStreamAudioSourceNode>> MediaStreamAudioSourceNode::create(BaseAudioContext& context, MediaStreamAudioSourceOptions&& options)
{
    RELEASE_ASSERT(options.mediaStream);

    auto audioTracks = options.mediaStream->getAudioTracks();
    if (audioTracks.isEmpty())
        return Exception { ExceptionCode::InvalidStateError, "Media stream has no audio tracks"_s };

    RefPtr<WebAudioSourceProvider> provider;
    for (auto& track : audioTracks) {
        provider = track->createAudioSourceProvider();
        if (provider)
            break;
    }
    if (!provider)
        return Exception { ExceptionCode::InvalidStateError, "Could not find an audio track with an audio source provider"_s };

    auto node = adoptRef(*new MediaStreamAudioSourceNode(context, *options.mediaStream, provider.releaseNonNull()));
    node->setFormat(2, context.sampleRate());

    // Context keeps reference until node is disconnected.
    context.sourceNodeWillBeginPlayback(node);

    return node;
}

MediaStreamAudioSourceNode::MediaStreamAudioSourceNode(BaseAudioContext& context, MediaStream& mediaStream, Ref<WebAudioSourceProvider>&& provider)
    : AudioNode(context, NodeTypeMediaStreamAudioSource)
    , m_mediaStream(mediaStream)
    , m_provider(provider)
{
    m_provider->setClient(this);
    
    // Default to stereo. This could change depending on the format of the MediaStream's audio track.
    addOutput(2);

    initialize();
}

MediaStreamAudioSourceNode::~MediaStreamAudioSourceNode()
{
    m_provider->setClient(nullptr);
    uninitialize();
}

void MediaStreamAudioSourceNode::setFormat(size_t numberOfChannels, float sourceSampleRate)
{
    // Synchronize with process().
    Locker locker { m_processLock };

    if (numberOfChannels == m_sourceNumberOfChannels && sourceSampleRate == m_sourceSampleRate)
        return;

    // The sample-rate must be equal to the context's sample-rate.
    if (!numberOfChannels || numberOfChannels > AudioContext::maxNumberOfChannels) {
        // process() will generate silence for these uninitialized values.
        LOG(Media, "MediaStreamAudioSourceNode::setFormat(%u, %f) - unhandled format change", static_cast<unsigned>(numberOfChannels), sourceSampleRate);
        m_sourceNumberOfChannels = 0;
        return;
    }

    m_sourceNumberOfChannels = numberOfChannels;
    m_sourceSampleRate = sourceSampleRate;

    float sampleRate = this->sampleRate();
    if (sourceSampleRate == sampleRate)
        m_multiChannelResampler = nullptr;
    else {
        double scaleFactor = sourceSampleRate / sampleRate;
        m_multiChannelResampler = makeUnique<MultiChannelResampler>(scaleFactor, numberOfChannels, AudioUtilities::renderQuantumSize, std::bind(&MediaStreamAudioSourceNode::provideInput, this, std::placeholders::_1, std::placeholders::_2));
    }

    m_sourceNumberOfChannels = numberOfChannels;

    {
        // The context must be locked when changing the number of output channels.
        Locker contextLocker { context().graphLock() };

        // Do any necesssary re-configuration to the output's number of channels.
        output(0)->setNumberOfChannels(numberOfChannels);
    }
}

void MediaStreamAudioSourceNode::provideInput(AudioBus* bus, size_t framesToProcess)
{
    m_provider->provideInput(bus, framesToProcess);
}

void MediaStreamAudioSourceNode::process(size_t numberOfFrames)
{
    AudioBus* outputBus = output(0)->bus();

    // Use tryLock() to avoid contention in the real-time audio thread.
    // If we fail to acquire the lock then the MediaStream must be in the middle of
    // a format change, so we output silence in this case.
    if (!m_processLock.tryLock()) {
        // We failed to acquire the lock.
        outputBus->zero();
        return;
    }
    Locker locker { AdoptLock, m_processLock };

    if (!m_sourceNumberOfChannels || !m_sourceSampleRate || m_sourceNumberOfChannels != outputBus->numberOfChannels()) {
        outputBus->zero();
        return;
    }

    if (numberOfFrames > outputBus->length())
        numberOfFrames = outputBus->length();

    if (m_multiChannelResampler) {
        ASSERT(m_sourceSampleRate != sampleRate());
        m_multiChannelResampler->process(outputBus, numberOfFrames);
    } else {
        // Bypass the resampler completely if the source is at the context's sample-rate.
        ASSERT(m_sourceSampleRate == sampleRate());
        provideInput(outputBus, numberOfFrames);
    }
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO) && ENABLE(MEDIA_STREAM)
