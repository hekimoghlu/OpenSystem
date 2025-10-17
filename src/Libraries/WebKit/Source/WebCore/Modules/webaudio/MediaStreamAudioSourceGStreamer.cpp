/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

#if ENABLE(MEDIA_STREAM) && USE(GSTREAMER) && ENABLE(WEB_AUDIO)

#include "AudioBus.h"
#include "GStreamerAudioData.h"
#include "GStreamerAudioStreamDescription.h"
#include "Logging.h"

namespace WebCore {

static void copyBusData(AudioBus& bus, GstBuffer* buffer, bool isMuted)
{
    GstMappedBuffer mappedBuffer(buffer, GST_MAP_WRITE);
    if (isMuted) {
        memset(mappedBuffer.data(), 0, mappedBuffer.size());
        return;
    }

    size_t offset = 0;
    for (size_t channelIndex = 0; channelIndex < bus.numberOfChannels(); ++channelIndex) {
        const auto& channel = *bus.channel(channelIndex);
        auto dataSize = sizeof(float) * channel.length();
        WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GLib port
        memcpy(mappedBuffer.data() + offset, channel.data(), dataSize);
        WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        offset += dataSize;
    }
}

void MediaStreamAudioSource::consumeAudio(AudioBus& bus, size_t numberOfFrames)
{
    if (!bus.numberOfChannels() || bus.numberOfChannels() > 2) {
        RELEASE_LOG_ERROR(Media, "MediaStreamAudioSource::consumeAudio(%p) trying to consume bus with %u channels", this, bus.numberOfChannels());
        return;
    }

    MediaTime mediaTime((m_numberOfFrames * G_USEC_PER_SEC) / m_currentSettings.sampleRate(), G_USEC_PER_SEC);
    m_numberOfFrames += numberOfFrames;

    // Lazily initialize caps, the settings don't change so this is OK.
    if (!m_caps || GST_AUDIO_INFO_CHANNELS(&m_info) != static_cast<int>(bus.numberOfChannels())) {
        gst_audio_info_set_format(&m_info, GST_AUDIO_FORMAT_F32LE, m_currentSettings.sampleRate(), bus.numberOfChannels(), nullptr);
        GST_AUDIO_INFO_LAYOUT(&m_info) = GST_AUDIO_LAYOUT_NON_INTERLEAVED;
        m_caps = adoptGRef(gst_audio_info_to_caps(&m_info));
    }

    size_t size = GST_AUDIO_INFO_BPS(&m_info) * bus.numberOfChannels() * numberOfFrames;
    auto buffer = adoptGRef(gst_buffer_new_allocate(nullptr, size, nullptr));

    GST_BUFFER_PTS(buffer.get()) = toGstClockTime(mediaTime);
    GST_BUFFER_FLAG_SET(buffer.get(), GST_BUFFER_FLAG_LIVE);

    copyBusData(bus, buffer.get(), muted());
    gst_buffer_add_audio_meta(buffer.get(), &m_info, numberOfFrames, nullptr);
    auto sample = adoptGRef(gst_sample_new(buffer.get(), m_caps.get(), nullptr, nullptr));
    GStreamerAudioData audioBuffer(WTFMove(sample), m_info);
    GStreamerAudioStreamDescription description(&m_info);
    audioSamplesAvailable(mediaTime, audioBuffer, description, numberOfFrames);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER) && ENABLE(WEB_AUDIO)
