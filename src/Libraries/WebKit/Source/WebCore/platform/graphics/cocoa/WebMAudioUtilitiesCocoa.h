/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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

#if PLATFORM(COCOA)

#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

struct AudioStreamBasicDescription;

namespace WebCore {

struct AudioInfo;
class SharedBuffer;

WEBCORE_EXPORT bool isVorbisDecoderAvailable();
WEBCORE_EXPORT bool registerVorbisDecoderIfNeeded();
static constexpr size_t kVorbisMinimumFrameDataSize = 1;
RefPtr<AudioInfo> createVorbisAudioInfo(std::span<const uint8_t>);

struct OpusCookieContents {
    uint8_t version { 0 };
    uint8_t channelCount { 0 };
    uint16_t preSkip { 0 };
    uint32_t sampleRate { 0 };
    int16_t outputGain { 0 };
    uint8_t mappingFamily { 0 };
    uint8_t config { 0 };
    Seconds frameDuration { 0_s };
    int32_t bandwidth { 0 };
    uint8_t framesPerPacket { 0 };
    bool isVBR { false };
    bool hasPadding { false };
#if HAVE(AUDIOFORMATPROPERTY_VARIABLEPACKET_SUPPORTED)
    RefPtr<SharedBuffer> cookieData;
#endif
};

WEBCORE_EXPORT bool isOpusDecoderAvailable();
WEBCORE_EXPORT bool registerOpusDecoderIfNeeded();
static constexpr size_t kOpusHeaderSize = 19;
static constexpr size_t kOpusMinimumFrameDataSize = 2;
std::optional<OpusCookieContents> parseOpusPrivateData(std::span<const uint8_t> privateData, std::span<const uint8_t> frameData);
bool parseOpusTOCData(std::span<const uint8_t> frameData, OpusCookieContents&);
RefPtr<AudioInfo> createOpusAudioInfo(const OpusCookieContents&);
Vector<uint8_t> createOpusPrivateData(const AudioStreamBasicDescription&, uint16_t preSkip = 0);

}

#endif // && PLATFORM(COCOA)
