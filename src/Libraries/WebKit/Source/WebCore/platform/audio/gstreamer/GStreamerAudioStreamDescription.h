/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#pragma once

#if USE(GSTREAMER)

#include "AudioStreamDescription.h"
#include "GStreamerCommon.h"
#include <gst/audio/audio.h>

namespace WebCore {

class GStreamerAudioStreamDescription final: public AudioStreamDescription {
public:
    GStreamerAudioStreamDescription(GstAudioInfo&& info)
        : m_info(WTFMove(info))
    {
    }

    GStreamerAudioStreamDescription(const GstAudioInfo& info)
        : m_info(info)
    {
    }

    GStreamerAudioStreamDescription(GstAudioInfo* info)
        : m_info(*info)
    {
    }

    GStreamerAudioStreamDescription()
    {
        gst_audio_info_init(&m_info);
    }

    ~GStreamerAudioStreamDescription() = default;

    const PlatformDescription& platformDescription() const
    {
        m_platformDescription = { PlatformDescription::GStreamerAudioStreamDescription, reinterpret_cast<const AudioStreamBasicDescription*>(&m_info) };

        return m_platformDescription;
    }

    PCMFormat format() const final
    {
        switch (GST_AUDIO_INFO_FORMAT(&m_info)) {
        case GST_AUDIO_FORMAT_S16LE:
        case GST_AUDIO_FORMAT_S16BE:
            return Int16;
        case GST_AUDIO_FORMAT_S32LE:
        case GST_AUDIO_FORMAT_S32BE:
            return Int32;
        case GST_AUDIO_FORMAT_F32LE:
        case GST_AUDIO_FORMAT_F32BE:
            return Float32;
        case GST_AUDIO_FORMAT_F64LE:
        case GST_AUDIO_FORMAT_F64BE:
            return Float64;
        default:
            break;
        }
        return None;
    }

    double sampleRate() const final { return GST_AUDIO_INFO_RATE(&m_info); }
    bool isPCM() const final { return format() != None; }
    bool isInterleaved() const final { return GST_AUDIO_INFO_LAYOUT(&m_info) == GST_AUDIO_LAYOUT_INTERLEAVED; }
    bool isSignedInteger() const final { return GST_AUDIO_INFO_IS_INTEGER(&m_info); }
    bool isNativeEndian() const final { return GST_AUDIO_INFO_ENDIANNESS(&m_info) == G_BYTE_ORDER; }
    bool isFloat() const final { return GST_AUDIO_INFO_IS_FLOAT(&m_info); }
    int bytesPerFrame() { return GST_AUDIO_INFO_BPF(&m_info);  }

    uint32_t numberOfInterleavedChannels() const final { return isInterleaved() ? GST_AUDIO_INFO_CHANNELS(&m_info) : TRUE; }
    uint32_t numberOfChannelStreams() const final { return GST_AUDIO_INFO_CHANNELS(&m_info); }
    uint32_t numberOfChannels() const final { return GST_AUDIO_INFO_CHANNELS(&m_info); }
    uint32_t sampleWordSize() const final { return GST_AUDIO_INFO_BPS(&m_info); }

    bool operator==(const GStreamerAudioStreamDescription& other) { return gst_audio_info_is_equal(&m_info, &other.m_info); }

    const GRefPtr<GstCaps>& caps()
    {
        if (!m_caps)
            m_caps = adoptGRef(gst_audio_info_to_caps(&m_info));
        return m_caps;
    }
    const GstAudioInfo& getInfo() const { return m_info; }

private:
    GstAudioInfo m_info;
    GRefPtr<GstCaps> m_caps;
    mutable PlatformDescription m_platformDescription;
};

} // WebCore

#endif // USE(GSTREAMER)
