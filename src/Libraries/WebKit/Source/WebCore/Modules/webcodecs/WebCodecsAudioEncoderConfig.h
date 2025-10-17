/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

#if ENABLE(WEB_CODECS)

#include "AacEncoderConfig.h"
#include "BitrateMode.h"
#include "FlacEncoderConfig.h"
#include "OpusEncoderConfig.h"
#include <optional>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct WebCodecsAudioEncoderConfig {
    String codec;
    size_t sampleRate;
    size_t numberOfChannels;
    std::optional<uint64_t> bitrate;
    BitrateMode bitrateMode { BitrateMode::Variable };
    std::optional<OpusEncoderConfig> opus;
    std::optional<AacEncoderConfig> aac;
    std::optional<FlacEncoderConfig> flac;

    WebCodecsAudioEncoderConfig isolatedCopy() && { return { WTFMove(codec).isolatedCopy(), sampleRate, numberOfChannels, bitrate, bitrateMode, opus, aac, flac }; }
    WebCodecsAudioEncoderConfig isolatedCopy() const & { return { codec.isolatedCopy(), sampleRate, numberOfChannels, bitrate, bitrateMode, opus, aac, flac }; }
};

}

#endif // ENABLE(WEB_CODECS)
