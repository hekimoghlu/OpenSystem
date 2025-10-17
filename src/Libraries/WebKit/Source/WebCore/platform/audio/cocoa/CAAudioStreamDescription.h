/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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

#include "AudioStreamDescription.h"
#include <CoreAudio/CoreAudioTypes.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

WEBCORE_EXPORT bool operator==(const AudioStreamBasicDescription&, const AudioStreamBasicDescription&);

class WEBCORE_EXPORT CAAudioStreamDescription final : public AudioStreamDescription {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(CAAudioStreamDescription, WEBCORE_EXPORT);
public:
    CAAudioStreamDescription(const AudioStreamBasicDescription&);
    enum class IsInterleaved : bool {
        No,
        Yes
    };
    CAAudioStreamDescription(double sampleRate, uint32_t channels, PCMFormat, IsInterleaved);
    ~CAAudioStreamDescription();

    const PlatformDescription& platformDescription() const final;

    PCMFormat format() const final;

    double sampleRate() const final;
    bool isPCM() const final;
    bool isInterleaved() const final;
    bool isSignedInteger() const final;
    bool isFloat() const final;
    bool isNativeEndian() const final;

    uint32_t numberOfInterleavedChannels() const final;
    uint32_t numberOfChannelStreams() const final;
    uint32_t numberOfChannels() const final;
    uint32_t sampleWordSize() const final;
    uint32_t bytesPerFrame() const;
    uint32_t bytesPerPacket() const;
    uint32_t formatFlags() const;

    bool operator==(const CAAudioStreamDescription& other) const { return operator==(static_cast<const AudioStreamDescription&>(other)); }
    bool operator==(const AudioStreamBasicDescription&) const;
    bool operator==(const AudioStreamDescription&) const;

    const AudioStreamBasicDescription& streamDescription() const;
    AudioStreamBasicDescription& streamDescription();

private:
    AudioStreamBasicDescription m_streamDescription;
    mutable PlatformDescription m_platformDescription;
    mutable PCMFormat m_format { None };
};

inline CAAudioStreamDescription toCAAudioStreamDescription(const AudioStreamDescription& description)
{
    ASSERT(description.platformDescription().type == PlatformDescription::CAAudioStreamBasicType);
    return CAAudioStreamDescription(*std::get<const AudioStreamBasicDescription*>(description.platformDescription().description));
}

}
