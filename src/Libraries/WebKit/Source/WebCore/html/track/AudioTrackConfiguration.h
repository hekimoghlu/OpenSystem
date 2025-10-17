/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

#if ENABLE(VIDEO)

#include "PlatformAudioTrackConfiguration.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

using AudioTrackConfigurationInit = PlatformAudioTrackConfiguration;

class AudioTrackConfiguration : public RefCounted<AudioTrackConfiguration> {
    WTF_MAKE_TZONE_ALLOCATED(AudioTrackConfiguration);
public:
    static Ref<AudioTrackConfiguration> create(AudioTrackConfigurationInit&& init) { return adoptRef(*new AudioTrackConfiguration(WTFMove(init))); }
    static Ref<AudioTrackConfiguration> create() { return adoptRef(*new AudioTrackConfiguration()); }

    void setState(const AudioTrackConfigurationInit& state) { m_state = state; }

    String codec() const { return m_state.codec; }
    void setCodec(String codec) { m_state.codec = codec; }

    uint32_t sampleRate() const { return m_state.sampleRate; }
    void setSampleRate(uint32_t sampleRate) { m_state.sampleRate = sampleRate; }

    uint32_t numberOfChannels() const { return m_state.numberOfChannels; }
    void setNumberOfChannels(uint32_t numberOfChannels) { m_state.numberOfChannels = numberOfChannels; }

    uint64_t bitrate() const { return m_state.bitrate; }
    void setBitrate(uint64_t bitrate) { m_state.bitrate = bitrate; }

    Ref<JSON::Object> toJSON() const;

private:
    AudioTrackConfiguration(AudioTrackConfigurationInit&& init)
        : m_state(init)
    {
    }
    AudioTrackConfiguration() = default;

    AudioTrackConfigurationInit m_state;
};

}

#endif
