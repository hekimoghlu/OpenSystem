/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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

#include <optional>
#include <wtf/Algorithms.h>
#include <wtf/Vector.h>

namespace WebCore {

enum class OpusBitstreamFormat : uint8_t {
    Opus,
    Ogg
};

// https://www.w3.org/TR/webcodecs-opus-codec-registration/#opus-encoder-config
struct OpusEncoderConfig {
    bool isValid()
    {
        float frameDurationMs = frameDuration / 1000.0;
        if (!WTF::anyOf(Vector<float> { 2.5, 5, 10, 20, 40, 60, 120 }, [frameDurationMs](auto value) -> bool {
            return WTF::areEssentiallyEqual(value, frameDurationMs);
        })) {
            return false;
        }
        if (complexity > 10)
            return false;

        if (packetlossperc > 100)
            return false;

        return true;
    }

    using BitstreamFormat = OpusBitstreamFormat;
    OpusBitstreamFormat format { OpusBitstreamFormat::Opus };
    uint64_t frameDuration { 20000 };
    size_t complexity { 9 };
    size_t packetlossperc { 0 };
    bool useinbandfec { false };
    bool usedtx { false };
};

}

#endif // ENABLE(WEB_CODECS)
