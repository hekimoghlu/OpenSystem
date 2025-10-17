/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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
#include "RealtimeOutgoingAudioSourceLibWebRTC.h"

#include "GStreamerAudioData.h"
#include "GStreamerAudioStreamDescription.h"
#include "LibWebRTCAudioFormat.h"
#include "LibWebRTCProvider.h"
#include "NotImplemented.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RealtimeOutgoingAudioSourceLibWebRTC);

RealtimeOutgoingAudioSourceLibWebRTC::RealtimeOutgoingAudioSourceLibWebRTC(Ref<MediaStreamTrackPrivate>&& audioSource)
    : RealtimeOutgoingAudioSource(WTFMove(audioSource))
{
    m_adapter = adoptGRef(gst_adapter_new()),
    m_sampleConverter = nullptr;
}

RealtimeOutgoingAudioSourceLibWebRTC::~RealtimeOutgoingAudioSourceLibWebRTC()
{
    m_sampleConverter = nullptr;
}

Ref<RealtimeOutgoingAudioSource> RealtimeOutgoingAudioSource::create(Ref<MediaStreamTrackPrivate>&& audioSource)
{
    return RealtimeOutgoingAudioSourceLibWebRTC::create(WTFMove(audioSource));
}

static inline GstAudioInfo libwebrtcAudioFormat(int sampleRate, size_t channelCount)
{
    GstAudioFormat format = gst_audio_format_build_integer(
        LibWebRTCAudioFormat::isSigned,
        LibWebRTCAudioFormat::isBigEndian ? G_BIG_ENDIAN : G_LITTLE_ENDIAN,
        LibWebRTCAudioFormat::sampleSize,
        LibWebRTCAudioFormat::sampleSize);

    GstAudioInfo info;

    size_t libWebRTCChannelCount = channelCount >= 2 ? 2 : channelCount;
    gst_audio_info_set_format(&info, format, sampleRate, libWebRTCChannelCount, nullptr);
    return info;
}

void RealtimeOutgoingAudioSourceLibWebRTC::audioSamplesAvailable(const MediaTime&, const PlatformAudioData& audioData, const AudioStreamDescription& streamDescription, size_t /* sampleCount */)
{
    DisableMallocRestrictionsForCurrentThreadScope disableMallocRestrictions;
    auto data = static_cast<const GStreamerAudioData&>(audioData);
    auto desc = static_cast<const GStreamerAudioStreamDescription&>(streamDescription);

    if (m_sampleConverter && !gst_audio_info_is_equal(&m_inputStreamDescription, &desc.getInfo())) {
        GST_ERROR_OBJECT(this, "FIXME - Audio format renegotiation is not possible yet!");
        m_sampleConverter = nullptr;
    }

    if (!m_sampleConverter) {
        m_inputStreamDescription = desc.getInfo();
        m_outputStreamDescription = libwebrtcAudioFormat(LibWebRTCAudioFormat::sampleRate, desc.numberOfChannels());
        m_sampleConverter.reset(gst_audio_converter_new(GST_AUDIO_CONVERTER_FLAG_IN_WRITABLE, &m_inputStreamDescription,
            &m_outputStreamDescription, nullptr));
    }

    {
        Locker locker { m_adapterLock };
        const auto& sample = data.getSample();
        auto* buffer = gst_sample_get_buffer(sample.get());
        gst_adapter_push(m_adapter.get(), gst_buffer_ref(buffer));
    }
    LibWebRTCProvider::callOnWebRTCSignalingThread([protectedThis = Ref { *this }] {
        protectedThis->pullAudioData();
    });
}

void RealtimeOutgoingAudioSourceLibWebRTC::pullAudioData()
{
    if (!GST_AUDIO_INFO_IS_VALID(&m_inputStreamDescription) || !GST_AUDIO_INFO_IS_VALID(&m_outputStreamDescription)) {
        GST_INFO("No stream description set yet.");
        return;
    }

    size_t outChunkSampleCount = LibWebRTCAudioFormat::chunkSampleCount;
    size_t outBufferSize = outChunkSampleCount * m_outputStreamDescription.bpf;

    Locker locker { m_adapterLock };
    size_t inChunkSampleCount = gst_audio_converter_get_in_frames(m_sampleConverter.get(), outChunkSampleCount);
    size_t inBufferSize = inChunkSampleCount * m_inputStreamDescription.bpf;

    while (gst_adapter_available(m_adapter.get()) > inBufferSize) {
        auto inBuffer = adoptGRef(gst_adapter_take_buffer(m_adapter.get(), inBufferSize));
        m_audioBuffer.grow(outBufferSize);
        if (isSilenced())
            webkitGstAudioFormatFillSilence(m_outputStreamDescription.finfo, m_audioBuffer.data(), outBufferSize);
        else {
            GstMappedBuffer inMap(inBuffer.get(), GST_MAP_READ);

            gpointer in[1] = { inMap.data() };
            gpointer out[1] = { m_audioBuffer.data() };
            if (!gst_audio_converter_samples(m_sampleConverter.get(), static_cast<GstAudioConverterFlags>(0), in, inChunkSampleCount, out, outChunkSampleCount)) {
                GST_ERROR("Could not convert samples.");

                return;
            }
        }

        sendAudioFrames(m_audioBuffer.data(), LibWebRTCAudioFormat::sampleSize, GST_AUDIO_INFO_RATE(&m_outputStreamDescription),
            GST_AUDIO_INFO_CHANNELS(&m_outputStreamDescription), outChunkSampleCount);
    }
}

bool RealtimeOutgoingAudioSourceLibWebRTC::isReachingBufferedAudioDataHighLimit()
{
    notImplemented();
    return false;
}

bool RealtimeOutgoingAudioSourceLibWebRTC::isReachingBufferedAudioDataLowLimit()
{
    notImplemented();
    return false;
}

bool RealtimeOutgoingAudioSourceLibWebRTC::hasBufferedEnoughData()
{
    notImplemented();
    return false;
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
