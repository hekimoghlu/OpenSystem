/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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

#include "GRefPtrGStreamer.h"
#include "PlatformAudioData.h"

#include <gst/audio/audio.h>

namespace WebCore {

class GStreamerAudioData final : public PlatformAudioData {
public:
    GStreamerAudioData(GRefPtr<GstSample>&& sample, GstAudioInfo&& info)
        : m_sample(WTFMove(sample))
        , m_audioInfo(WTFMove(info))
    {
    }

    GStreamerAudioData(GRefPtr<GstSample>&& sample, const GstAudioInfo& info)
        : m_sample(WTFMove(sample))
        , m_audioInfo(info)
    {
    }

    GStreamerAudioData(GRefPtr<GstSample>&& sample)
        : m_sample(WTFMove(sample))
    {
        gst_audio_info_from_caps(&m_audioInfo, gst_sample_get_caps(m_sample.get()));
    }

    void setSample(GRefPtr<GstSample>&& sample) { m_sample = WTFMove(sample); }
    const GRefPtr<GstSample>& getSample() const { return m_sample; }
    const GstAudioInfo& getAudioInfo() const { return m_audioInfo; }
    uint32_t channelCount() const { return GST_AUDIO_INFO_CHANNELS(&m_audioInfo); }

private:
    Kind kind() const { return Kind::GStreamerAudioData; }
    GRefPtr<GstSample> m_sample;
    GstAudioInfo m_audioInfo;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::GStreamerAudioData)
static bool isType(const WebCore::PlatformAudioData& data) { return data.kind() == WebCore::PlatformAudioData::Kind::GStreamerAudioData; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(GSTREAMER)
