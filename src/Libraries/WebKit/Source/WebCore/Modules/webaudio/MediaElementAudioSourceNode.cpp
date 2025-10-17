/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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

#if ENABLE(WEB_AUDIO) && ENABLE(VIDEO)

#include "MediaElementAudioSourceNode.h"

#include "AudioContext.h"
#include "AudioNodeOutput.h"
#include "AudioSourceProvider.h"
#include "AudioUtilities.h"
#include "Logging.h"
#include "MediaElementAudioSourceOptions.h"
#include "MediaPlayer.h"
#include "SecurityOrigin.h"
#include <wtf/Locker.h>
#include <wtf/TZoneMallocInlines.h>

// These are somewhat arbitrary limits, but we need to do some kind of sanity-checking.
constexpr unsigned minSampleRate = 8000;
constexpr unsigned maxSampleRate = 192000;

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MediaElementAudioSourceNode);

ExceptionOr<Ref<MediaElementAudioSourceNode>> MediaElementAudioSourceNode::create(BaseAudioContext& context, MediaElementAudioSourceOptions&& options)
{
    RELEASE_ASSERT(options.mediaElement);

    if (options.mediaElement->audioSourceNode())
        return Exception { ExceptionCode::InvalidStateError, "Media element is already associated with an audio source node"_s };

    auto node = adoptRef(*new MediaElementAudioSourceNode(context, *options.mediaElement));

    options.mediaElement->setAudioSourceNode(node.ptr());

    // context keeps reference until node is disconnected.
    context.sourceNodeWillBeginPlayback(node);

    return node;
}

MediaElementAudioSourceNode::MediaElementAudioSourceNode(BaseAudioContext& context, Ref<HTMLMediaElement>&& mediaElement)
    : AudioNode(context, NodeTypeMediaElementAudioSource)
    , m_mediaElement(WTFMove(mediaElement))
{
    // Default to stereo. This could change depending on what the media element .src is set to.
    addOutput(2);

    initialize();
}

MediaElementAudioSourceNode::~MediaElementAudioSourceNode()
{
    m_mediaElement->setAudioSourceNode(nullptr);
    uninitialize();
}

void MediaElementAudioSourceNode::setFormat(size_t numberOfChannels, float sourceSampleRate)
{
    Ref protectedThis { *this };

    // Synchronize with process().
    Locker locker { m_processLock };

    m_muted = wouldTaintOrigin();

    if (numberOfChannels != m_sourceNumberOfChannels || sourceSampleRate != m_sourceSampleRate) {
        if (!numberOfChannels || numberOfChannels > AudioContext::maxNumberOfChannels || sourceSampleRate < minSampleRate || sourceSampleRate > maxSampleRate) {
            // process() will generate silence for these uninitialized values.
            LOG(Media, "MediaElementAudioSourceNode::setFormat(%u, %f) - unhandled format change", static_cast<unsigned>(numberOfChannels), sourceSampleRate);
            m_sourceNumberOfChannels = 0;
            m_sourceSampleRate = 0;
            return;
        }

        m_sourceNumberOfChannels = numberOfChannels;
        m_sourceSampleRate = sourceSampleRate;

        if (sourceSampleRate != sampleRate()) {
            double scaleFactor = sourceSampleRate / sampleRate();
            m_multiChannelResampler = makeUnique<MultiChannelResampler>(scaleFactor, numberOfChannels, AudioUtilities::renderQuantumSize, std::bind(&MediaElementAudioSourceNode::provideInput, this, std::placeholders::_1, std::placeholders::_2));
        } else {
            // Bypass resampling.
            m_multiChannelResampler = nullptr;
        }

        {
            // The context must be locked when changing the number of output channels.
            Locker contextLocker { context().graphLock() };

            // Do any necesssary re-configuration to the output's number of channels.
            output(0)->setNumberOfChannels(numberOfChannels);
        }
    }
}

void MediaElementAudioSourceNode::provideInput(AudioBus* bus, size_t framesToProcess)
{
    ASSERT(bus);
    if (auto* provider = mediaElement().audioSourceProvider())
        provider->provideInput(bus, framesToProcess);
    else {
        // Either this port doesn't yet support HTMLMediaElement audio stream access,
        // or the stream is not yet available.
        bus->zero();
    }
}

bool MediaElementAudioSourceNode::wouldTaintOrigin()
{
    if (RefPtr origin = context().origin())
        return m_mediaElement->taintsOrigin(*origin);

    return true;
}

void MediaElementAudioSourceNode::process(size_t numberOfFrames)
{
    AudioBus* outputBus = output(0)->bus();

    // Use tryLock() to avoid contention in the real-time audio thread.
    // If we fail to acquire the lock then the HTMLMediaElement must be in the middle of
    // reconfiguring its playback engine, so we output silence in this case.
    if (!m_processLock.tryLock()) {
        // We failed to acquire the lock.
        outputBus->zero();
        return;
    }

    Locker locker { AdoptLock, m_processLock };

    if (m_muted || !m_sourceNumberOfChannels || !m_sourceSampleRate || m_sourceNumberOfChannels != outputBus->numberOfChannels()) {
        outputBus->zero();
        return;
    }

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

#endif // ENABLE(WEB_AUDIO)
