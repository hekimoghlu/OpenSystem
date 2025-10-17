/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#include "RealtimeOutgoingAudioSourceCocoa.h"

#if USE(LIBWEBRTC)

#include "CAAudioStreamDescription.h"
#include "LibWebRTCAudioFormat.h"
#include "LibWebRTCProvider.h"
#include "Logging.h"
#include "SpanCoreAudio.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RealtimeOutgoingAudioSourceCocoa);

static inline AudioStreamBasicDescription libwebrtcAudioFormat(Float64 sampleRate, size_t channelCount)
{
    // FIXME: Microphones can have more than two channels. In such case, we should do the mix down based on AudioChannelLayoutTag.
    size_t libWebRTCChannelCount = channelCount >= 2 ? 2 : channelCount;
    AudioStreamBasicDescription streamFormat;
    FillOutASBDForLPCM(streamFormat, sampleRate, libWebRTCChannelCount, LibWebRTCAudioFormat::sampleSize, LibWebRTCAudioFormat::sampleSize, LibWebRTCAudioFormat::isFloat, LibWebRTCAudioFormat::isBigEndian, LibWebRTCAudioFormat::isNonInterleaved);
    return streamFormat;
}

RealtimeOutgoingAudioSourceCocoa::RealtimeOutgoingAudioSourceCocoa(Ref<MediaStreamTrackPrivate>&& audioSource)
    : RealtimeOutgoingAudioSource(WTFMove(audioSource))
    , m_sampleConverter(AudioSampleDataSource::create(LibWebRTCAudioFormat::sampleRate * 2, source()))
{
}

RealtimeOutgoingAudioSourceCocoa::~RealtimeOutgoingAudioSourceCocoa() = default;

Ref<RealtimeOutgoingAudioSource> RealtimeOutgoingAudioSource::create(Ref<MediaStreamTrackPrivate>&& audioSource)
{
    return RealtimeOutgoingAudioSourceCocoa::create(WTFMove(audioSource));
}

bool RealtimeOutgoingAudioSourceCocoa::isReachingBufferedAudioDataHighLimit()
{
    auto writtenAudioDuration = m_writeCount / m_inputStreamDescription->sampleRate();
    auto readAudioDuration = m_readCount / m_outputStreamDescription->sampleRate();

    ASSERT(writtenAudioDuration >= readAudioDuration);
    return writtenAudioDuration > readAudioDuration + 0.5;
}

bool RealtimeOutgoingAudioSourceCocoa::isReachingBufferedAudioDataLowLimit()
{
    auto writtenAudioDuration = m_writeCount / m_inputStreamDescription->sampleRate();
    auto readAudioDuration = m_readCount / m_outputStreamDescription->sampleRate();

    ASSERT(writtenAudioDuration >= readAudioDuration);
    return writtenAudioDuration < readAudioDuration + 0.1;
}

bool RealtimeOutgoingAudioSourceCocoa::hasBufferedEnoughData()
{
    auto writtenAudioDuration = m_writeCount / m_inputStreamDescription->sampleRate();
    auto readAudioDuration = m_readCount / m_outputStreamDescription->sampleRate();

    ASSERT(writtenAudioDuration >= readAudioDuration);
    return writtenAudioDuration >= readAudioDuration + 0.01;
}

// May get called on a background thread.
void RealtimeOutgoingAudioSourceCocoa::audioSamplesAvailable(const MediaTime&, const PlatformAudioData& audioData, const AudioStreamDescription& streamDescription, size_t sampleCount)
{
    if (m_inputStreamDescription != streamDescription) {
        if (m_writeCount && m_inputStreamDescription->sampleRate()) {
            // Update m_writeCount to be valid according the new sampleRate.
            m_writeCount = (m_writeCount * streamDescription.sampleRate()) / m_inputStreamDescription->sampleRate();
        }

        m_inputStreamDescription = toCAAudioStreamDescription(streamDescription);
        auto status  = m_sampleConverter->setInputFormat(*m_inputStreamDescription);
        ASSERT_UNUSED(status, !status);

        m_outputStreamDescription = libwebrtcAudioFormat(LibWebRTCAudioFormat::sampleRate, streamDescription.numberOfChannels());
        status = m_sampleConverter->setOutputFormat(m_outputStreamDescription->streamDescription());
        ASSERT(!status);
    }

    // Let's skip pushing samples if we are too slow pulling them.
    if (m_skippingAudioData) {
        if (!isReachingBufferedAudioDataLowLimit())
            return;
        m_skippingAudioData = false;
    } else if (isReachingBufferedAudioDataHighLimit()) {
        m_skippingAudioData = true;
        return;
    }

    m_sampleConverter->pushSamples(MediaTime(m_writeCount, static_cast<uint32_t>(m_inputStreamDescription->sampleRate())), audioData, sampleCount);
    m_writeCount += sampleCount;

    if (!hasBufferedEnoughData())
        return;

    // Heap allocations are forbidden on the audio thread for performance reasons so we need to
    // explicitly allow the following allocation(s).
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
    LibWebRTCProvider::callOnWebRTCSignalingThread([protectedThis = Ref { *this }] {
        protectedThis->pullAudioData();
    });
}

void RealtimeOutgoingAudioSourceCocoa::pullAudioData()
{
    // libwebrtc expects 10 ms chunks.
    size_t chunkSampleCount = m_outputStreamDescription->sampleRate() / 100;
    size_t bufferSize = chunkSampleCount * LibWebRTCAudioFormat::sampleByteSize * m_outputStreamDescription->numberOfChannels();
    m_audioBuffer.grow(bufferSize);

    AudioBufferList bufferList;
    bufferList.mNumberBuffers = 1;
    auto& firstBuffer = span(bufferList)[0];
    firstBuffer.mNumberChannels = m_outputStreamDescription->numberOfChannels();
    firstBuffer.mDataByteSize = bufferSize;
    firstBuffer.mData = m_audioBuffer.data();

    if (isSilenced() !=  m_sampleConverter->muted())
        m_sampleConverter->setMuted(isSilenced());

    m_sampleConverter->pullAvailableSamplesAsChunks(bufferList, chunkSampleCount, m_readCount, [this, chunkSampleCount] {
        m_readCount += chunkSampleCount;
        sendAudioFrames(m_audioBuffer.data(), LibWebRTCAudioFormat::sampleSize, m_outputStreamDescription->sampleRate(), m_outputStreamDescription->numberOfChannels(), chunkSampleCount);
    });
}

void RealtimeOutgoingAudioSourceCocoa::sourceUpdated()
{
#if !RELEASE_LOG_DISABLED
    m_sampleConverter->setLogger(source().logger(), source().logIdentifier());
#endif
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
