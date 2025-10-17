/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
#include "RealtimeIncomingAudioSourceCocoa.h"

#if USE(LIBWEBRTC)

#include "AudioStreamDescription.h"
#include "CAAudioStreamDescription.h"
#include "LibWebRTCAudioFormat.h"
#include "LibWebRTCAudioModule.h"
#include "Logging.h"
#include <pal/avfoundation/MediaTimeAVFoundation.h>

#include <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

Ref<RealtimeIncomingAudioSource> RealtimeIncomingAudioSource::create(rtc::scoped_refptr<webrtc::AudioTrackInterface>&& audioTrack, String&& audioTrackId)
{
    auto source = RealtimeIncomingAudioSourceCocoa::create(WTFMove(audioTrack), WTFMove(audioTrackId));
    source->start();
    return WTFMove(source);
}

Ref<RealtimeIncomingAudioSourceCocoa> RealtimeIncomingAudioSourceCocoa::create(rtc::scoped_refptr<webrtc::AudioTrackInterface>&& audioTrack, String&& audioTrackId)
{
    return adoptRef(*new RealtimeIncomingAudioSourceCocoa(WTFMove(audioTrack), WTFMove(audioTrackId)));
}

static inline AudioStreamBasicDescription streamDescription(size_t sampleRate, size_t channelCount)
{
    AudioStreamBasicDescription streamFormat;
    FillOutASBDForLPCM(streamFormat, sampleRate, channelCount, LibWebRTCAudioFormat::sampleSize, LibWebRTCAudioFormat::sampleSize, LibWebRTCAudioFormat::isFloat, LibWebRTCAudioFormat::isBigEndian, LibWebRTCAudioFormat::isNonInterleaved);
    return streamFormat;
}

RealtimeIncomingAudioSourceCocoa::RealtimeIncomingAudioSourceCocoa(rtc::scoped_refptr<webrtc::AudioTrackInterface>&& audioTrack, String&& audioTrackId)
    : RealtimeIncomingAudioSource(WTFMove(audioTrack), WTFMove(audioTrackId))
    , m_sampleRate(LibWebRTCAudioFormat::sampleRate)
    , m_numberOfChannels(1)
    , m_streamDescription(streamDescription(m_sampleRate, m_numberOfChannels))
    , m_audioBufferList(makeUnique<WebAudioBufferList>(m_streamDescription))
#if !RELEASE_LOG_DISABLED
    , m_logTimer(*this, &RealtimeIncomingAudioSourceCocoa::logTimerFired)
#endif
{
}

void RealtimeIncomingAudioSourceCocoa::startProducingData()
{
    RealtimeIncomingAudioSource::startProducingData();
#if !RELEASE_LOG_DISABLED
    m_logTimer.startRepeating(LogTimerInterval);
#endif
}

void RealtimeIncomingAudioSourceCocoa::stopProducingData()
{
#if !RELEASE_LOG_DISABLED
    m_logTimer.stop();
#endif
    RealtimeIncomingAudioSource::stopProducingData();
}

#if !RELEASE_LOG_DISABLED
void RealtimeIncomingAudioSourceCocoa::logTimerFired()
{
    if (!m_lastChunksReceived || (m_chunksReceived - m_lastChunksReceived) >= ChunksReceivedCountForLogging) {
        m_lastChunksReceived = m_chunksReceived;
        ALWAYS_LOG_IF(loggerPtr(), LOGIDENTIFIER, "chunk ", m_chunksReceived);
    }
    if (m_audioFormatChanged) {
        ALWAYS_LOG_IF(loggerPtr(), LOGIDENTIFIER, "new audio buffer list for sampleRate ", m_sampleRate, " and ", m_numberOfChannels, " channel(s)");
        m_audioFormatChanged = false;
    }
}
#endif

void RealtimeIncomingAudioSourceCocoa::OnData(const void* audioData, int bitsPerSample, int sampleRate, size_t numberOfChannels, size_t numberOfFrames)
{
    ++m_chunksReceived;

    static constexpr size_t initialSampleRate = 16000;
    static constexpr size_t initialChunksReceived = 20;
    // We usually receive some initial callbacks with no data at 16000, then we got real data at the actual sample rate.
    // To limit reallocations, let's skip these initial calls.
    if (m_chunksReceived < initialChunksReceived && sampleRate == initialSampleRate)
        return;

    if (!m_audioBufferList || m_numberOfChannels != numberOfChannels || m_sampleRate != sampleRate) {
#if !RELEASE_LOG_DISABLED
        m_audioFormatChanged = true;
#endif

        m_sampleRate = sampleRate;
        m_numberOfChannels = numberOfChannels;
        m_streamDescription = streamDescription(sampleRate, numberOfChannels);

        {
            DisableMallocRestrictionsForCurrentThreadScope scope;
            m_audioBufferList = makeUnique<WebAudioBufferList>(m_streamDescription);
        }
        if (m_sampleRate && m_numberOfFrames)
            m_numberOfFrames = m_numberOfFrames * sampleRate / m_sampleRate;
        else
            m_numberOfFrames = 0;
    }

    CMTime startTime = PAL::CMTimeMake(audioModule() ? audioModule()->currentAudioSampleCount() : m_numberOfFrames, LibWebRTCAudioFormat::sampleRate);
    auto mediaTime = PAL::toMediaTime(startTime);
    m_numberOfFrames += numberOfFrames;

    auto& bufferList = *m_audioBufferList->buffer(0);
    bufferList.mDataByteSize = numberOfChannels * numberOfFrames * bitsPerSample / 8;
    bufferList.mNumberChannels = numberOfChannels;
    bufferList.mData = const_cast<void*>(audioData);

    audioSamplesAvailable(mediaTime, *m_audioBufferList, m_streamDescription, numberOfFrames);
}

}

#endif // USE(LIBWEBRTC)
