/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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

#include "PlatformRawAudioData.h"

#if ENABLE(WEB_CODECS) && USE(GSTREAMER)

#include "GRefPtrGStreamer.h"
#include <gst/audio/audio.h>

namespace WebCore {

class PlatformRawAudioDataGStreamer final : public PlatformRawAudioData {
public:
    static Ref<PlatformRawAudioData> create(GRefPtr<GstSample>&& sample)
    {
        return adoptRef(*new PlatformRawAudioDataGStreamer(WTFMove(sample)));
    }

    AudioSampleFormat format() const final;
    size_t sampleRate() const final;
    size_t numberOfChannels() const final;
    size_t numberOfFrames() const final;
    std::optional<uint64_t> duration() const final;
    int64_t timestamp() const final;

    size_t memoryCost() const final;

    constexpr MediaPlatformType platformType() const final { return MediaPlatformType::GStreamer; }

    GstSample* sample() const { return m_sample.get(); }
    const GstAudioInfo* info() const { return &m_info; }

private:
    PlatformRawAudioDataGStreamer(GRefPtr<GstSample>&&);

    GRefPtr<GstSample> m_sample;
    GstAudioInfo m_info;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PlatformRawAudioDataGStreamer)
static bool isType(const WebCore::PlatformRawAudioData& data) { return data.platformType() == WebCore::MediaPlatformType::GStreamer; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEB_CODECS) && USE(GSTREAMER)
