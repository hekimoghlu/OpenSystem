/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
#include "RealtimeOutgoingAudioSource.h"

#if USE(LIBWEBRTC)

#include "LibWebRTCAudioFormat.h"
#include "LibWebRTCProvider.h"
#include "Logging.h"

namespace WebCore {

RealtimeOutgoingAudioSource::RealtimeOutgoingAudioSource(Ref<MediaStreamTrackPrivate>&& source)
    : m_audioSource(WTFMove(source))
{
}

RealtimeOutgoingAudioSource::~RealtimeOutgoingAudioSource()
{
    ASSERT(!m_audioSource->hasObserver(*this));
#if ASSERT_ENABLED
    Locker locker { m_sinksLock };
#endif
    ASSERT(m_sinks.isEmpty());

    stop();
}

void RealtimeOutgoingAudioSource::observeSource()
{
    ASSERT(!m_audioSource->hasObserver(*this));
    m_audioSource->addObserver(*this);
    m_audioSource->source().addAudioSampleObserver(*this);
    initializeConverter();
}

void RealtimeOutgoingAudioSource::unobserveSource()
{
    m_audioSource->source().removeAudioSampleObserver(*this);
    m_audioSource->removeObserver(*this);
}

void RealtimeOutgoingAudioSource::setSource(Ref<MediaStreamTrackPrivate>&& newSource)
{
    ALWAYS_LOG("Changing source to ", newSource->logIdentifier());

    ASSERT(!m_audioSource->hasObserver(*this));
    m_audioSource = WTFMove(newSource);
    sourceUpdated();
}

void RealtimeOutgoingAudioSource::initializeConverter()
{
    m_muted = m_audioSource->muted();
    m_enabled = m_audioSource->enabled();
}

void RealtimeOutgoingAudioSource::sourceMutedChanged()
{
    m_muted = m_audioSource->muted();
}

void RealtimeOutgoingAudioSource::sourceEnabledChanged()
{
    m_enabled = m_audioSource->enabled();
}

void RealtimeOutgoingAudioSource::AddSink(webrtc::AudioTrackSinkInterface* sink)
{
    Locker locker { m_sinksLock };
    m_sinks.add(sink);
}

void RealtimeOutgoingAudioSource::RemoveSink(webrtc::AudioTrackSinkInterface* sink)
{
    Locker locker { m_sinksLock };
    m_sinks.remove(sink);
}

void RealtimeOutgoingAudioSource::sendAudioFrames(const void* audioData, int bitsPerSample, int sampleRate, size_t numberOfChannels, size_t numberOfFrames)
{
#if !RELEASE_LOG_DISABLED
    if (!(++m_chunksSent % 200))
        ALWAYS_LOG(LOGIDENTIFIER, "chunk ", m_chunksSent);
#endif

    Locker locker { m_sinksLock };
    for (auto sink : m_sinks)
        sink->OnData(audioData, bitsPerSample, sampleRate, numberOfChannels, numberOfFrames);
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& RealtimeOutgoingAudioSource::logChannel() const
{
    return LogWebRTC;
}
#endif
    
} // namespace WebCore

#endif // USE(LIBWEBRTC)
