/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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

#include <variant>

typedef struct AudioStreamBasicDescription AudioStreamBasicDescription;

namespace WebCore {

struct PlatformDescription {
    enum {
        None,
        CAAudioStreamBasicType,
        GStreamerAudioStreamDescription,
    } type { None };
    std::variant<std::nullptr_t, const AudioStreamBasicDescription*> description;
};

class AudioStreamDescription {
public:
    virtual ~AudioStreamDescription() = default;

    virtual const PlatformDescription& platformDescription() const = 0;

    enum PCMFormat {
        None,
        Uint8,
        Int16,
        Int24,
        Int32,
        Float32,
        Float64
    };
    virtual PCMFormat format() const = 0;

    virtual double sampleRate() const = 0;

    virtual bool isPCM() const { return format() != None; }
    virtual bool isInterleaved() const = 0;
    virtual bool isSignedInteger() const = 0;
    virtual bool isFloat() const = 0;
    virtual bool isNativeEndian() const = 0;

    virtual uint32_t numberOfInterleavedChannels() const = 0;
    virtual uint32_t numberOfChannelStreams() const = 0;
    virtual uint32_t numberOfChannels() const = 0;
    virtual uint32_t sampleWordSize() const = 0;
};

}
