/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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

#if USE(LIBWEBRTC) && USE(GSTREAMER)
#include "RealtimeIncomingAudioSourceLibWebRTC.h"

#include "LibWebRTCAudioFormat.h"
#include "gstreamer/GStreamerAudioData.h"
#include "gstreamer/GStreamerAudioStreamDescription.h"

namespace WebCore {

Ref<RealtimeIncomingAudioSource> RealtimeIncomingAudioSource::create(rtc::scoped_refptr<webrtc::AudioTrackInterface>&& audioTrack, String&& audioTrackId)
{
    auto source = RealtimeIncomingAudioSourceLibWebRTC::create(WTFMove(audioTrack), WTFMove(audioTrackId));
    source->start();
    return source;
}

Ref<RealtimeIncomingAudioSourceLibWebRTC> RealtimeIncomingAudioSourceLibWebRTC::create(rtc::scoped_refptr<webrtc::AudioTrackInterface>&& audioTrack, String&& audioTrackId)
{
    return adoptRef(*new RealtimeIncomingAudioSourceLibWebRTC(WTFMove(audioTrack), WTFMove(audioTrackId)));
}

RealtimeIncomingAudioSourceLibWebRTC::RealtimeIncomingAudioSourceLibWebRTC(rtc::scoped_refptr<webrtc::AudioTrackInterface>&& audioTrack, String&& audioTrackId)
    : RealtimeIncomingAudioSource(WTFMove(audioTrack), WTFMove(audioTrackId))
{
}

void RealtimeIncomingAudioSourceLibWebRTC::OnData(const void* audioData, int, int sampleRate, size_t numberOfChannels, size_t numberOfFrames)
{
    GstAudioInfo info;
    GstAudioFormat format = gst_audio_format_build_integer(
        LibWebRTCAudioFormat::isSigned,
        LibWebRTCAudioFormat::isBigEndian ? G_BIG_ENDIAN : G_LITTLE_ENDIAN,
        LibWebRTCAudioFormat::sampleSize,
        LibWebRTCAudioFormat::sampleSize);

    gst_audio_info_set_format(&info, format, sampleRate, numberOfChannels, NULL);

    auto bufferSize = GST_AUDIO_INFO_BPF(&info) * numberOfFrames;
    auto buffer = adoptGRef(gst_buffer_new_memdup(const_cast<gpointer>(audioData), bufferSize));
    gst_buffer_add_audio_meta(buffer.get(), &info, numberOfFrames, nullptr);
    auto caps = adoptGRef(gst_audio_info_to_caps(&info));

    if (m_baseTime == MediaTime::invalidTime())
        m_baseTime = MediaTime::createWithSeconds(MonotonicTime::now().secondsSinceEpoch());

    MediaTime mediaTime = m_baseTime + MediaTime((m_numberOfFrames * G_USEC_PER_SEC) / sampleRate, G_USEC_PER_SEC);
    GST_BUFFER_PTS(buffer.get()) = toGstUnsigned64Time(mediaTime);

    auto sample = adoptGRef(gst_sample_new(buffer.get(), caps.get(), nullptr, nullptr));
    GStreamerAudioData data(WTFMove(sample), info);
    audioSamplesAvailable(mediaTime, data, GStreamerAudioStreamDescription(info), numberOfFrames);

    m_numberOfFrames += numberOfFrames;
}
}

#endif // USE(LIBWEBRTC)
