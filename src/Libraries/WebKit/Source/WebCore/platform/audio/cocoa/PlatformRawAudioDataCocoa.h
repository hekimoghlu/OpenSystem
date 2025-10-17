/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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

#include "CAAudioStreamDescription.h"
#include "PlatformRawAudioData.h"
#include <wtf/Forward.h>

#if ENABLE(WEB_CODECS) && USE(AVFOUNDATION)

typedef struct opaqueCMSampleBuffer *CMSampleBufferRef;
struct AudioStreamBasicDescription;

namespace WebCore {

class MediaSampleAVFObjC;

class PlatformRawAudioDataCocoa final : public PlatformRawAudioData {
public:
    static Ref<PlatformRawAudioData> create(Ref<MediaSampleAVFObjC>&& sample)
    {
        return adoptRef(*new PlatformRawAudioDataCocoa(WTFMove(sample)));
    }
    AudioSampleFormat format() const final;
    size_t sampleRate() const final;
    size_t numberOfChannels() const final;
    size_t numberOfFrames() const final;
    std::optional<uint64_t> duration() const final;
    int64_t timestamp() const final;
    size_t memoryCost() const final;

    constexpr MediaPlatformType platformType() const final { return MediaPlatformType::AVFObjC; }

    const CAAudioStreamDescription& description() const;
    CMSampleBufferRef sampleBuffer() const;

private:
    friend class PlatformRawAudioData;
    PlatformRawAudioDataCocoa(Ref<MediaSampleAVFObjC>&&);
    const AudioStreamBasicDescription& asbd() const;
    bool isInterleaved() const;

    const Ref<MediaSampleAVFObjC> m_sample;
    const CAAudioStreamDescription m_description;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PlatformRawAudioDataCocoa)
static bool isType(const WebCore::PlatformRawAudioData& data) { return data.platformType() == WebCore::MediaPlatformType::AVFObjC; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEB_CODECS) && USE(GSTREAMER)
