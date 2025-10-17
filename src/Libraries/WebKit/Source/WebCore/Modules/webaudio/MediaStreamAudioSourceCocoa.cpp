/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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
#include "MediaStreamAudioSource.h"

#if ENABLE(MEDIA_STREAM)

#include "AudioBus.h"
#include "CAAudioStreamDescription.h"
#include "Logging.h"
#include "SpanCoreAudio.h"
#include "WebAudioBufferList.h"
#include <CoreAudio/CoreAudioTypes.h>
#include <pal/avfoundation/MediaTimeAVFoundation.h>
#include <wtf/StdLibExtras.h>

#include <pal/cf/CoreMediaSoftLink.h>
#include "CoreVideoSoftLink.h"

namespace WebCore {

static inline CAAudioStreamDescription streamDescription(size_t sampleRate, size_t channelCount)
{
    bool isFloat = true;
    bool isBigEndian = false;
    bool isNonInterleaved = true;
    static const size_t sampleSize = 8 * sizeof(float);

    AudioStreamBasicDescription streamFormat;
    FillOutASBDForLPCM(streamFormat, sampleRate, channelCount, sampleSize, sampleSize, isFloat, isBigEndian, isNonInterleaved);
    return streamFormat;
}

static inline void copyChannelData(AudioChannel& channel, AudioBuffer& buffer, size_t numberOfFrames, bool isMuted)
{
    buffer.mDataByteSize = numberOfFrames * sizeof(float);
    buffer.mNumberChannels = 1;
    if (isMuted) {
        zeroSpan(mutableSpan<uint8_t>(buffer));
        return;
    }
    memcpySpan(mutableSpan<uint8_t>(buffer), asByteSpan(channel.span()).first(buffer.mDataByteSize));
}

void MediaStreamAudioSource::consumeAudio(AudioBus& bus, size_t numberOfFrames)
{
    if (bus.numberOfChannels() != 1 && bus.numberOfChannels() != 2) {
        RELEASE_LOG_ERROR(Media, "MediaStreamAudioSource::consumeAudio(%p) trying to consume bus with %u channels", this, bus.numberOfChannels());
        return;
    }

    CMTime startTime = PAL::CMTimeMake(m_numberOfFrames, m_currentSettings.sampleRate());
    auto mediaTime = PAL::toMediaTime(startTime);
    m_numberOfFrames += numberOfFrames;

    auto* audioBuffer = m_audioBuffer ? &downcast<WebAudioBufferList>(*m_audioBuffer) : nullptr;

    auto description = streamDescription(m_currentSettings.sampleRate(), bus.numberOfChannels());
    if (!audioBuffer || audioBuffer->channelCount() != bus.numberOfChannels()) {
        // Heap allocations are forbidden on the audio thread for performance reasons so we need to
        // explicitly allow the following allocation(s).
        DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
        m_audioBuffer = makeUnique<WebAudioBufferList>(description, numberOfFrames);
        audioBuffer = &downcast<WebAudioBufferList>(*m_audioBuffer);
    } else
        audioBuffer->setSampleCount(numberOfFrames);

    for (size_t cptr = 0; cptr < bus.numberOfChannels(); ++cptr)
        copyChannelData(*bus.channel(cptr), *audioBuffer->buffer(cptr), numberOfFrames, muted());

    audioSamplesAvailable(mediaTime, *m_audioBuffer, description, numberOfFrames);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
