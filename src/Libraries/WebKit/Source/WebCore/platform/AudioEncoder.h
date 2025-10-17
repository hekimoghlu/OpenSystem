/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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

#if ENABLE(WEB_CODECS)

#include "AudioEncoderActiveConfiguration.h"
#include "BitrateMode.h"
#include "WebCodecsAudioInternalData.h"
#include <span>
#include <wtf/CompletionHandler.h>
#include <wtf/NativePromise.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioEncoder : public ThreadSafeRefCounted<AudioEncoder> {
public:
    virtual ~AudioEncoder() = default;

    struct OpusConfig {
        bool isOggBitStream { false };
        uint64_t frameDuration { 20000 };
        std::optional<size_t> complexity { };
        size_t packetlossperc { 0 };
        bool useinbandfec { false };
        bool usedtx { false };
    };

    struct FlacConfig {
        size_t blockSize;
        size_t compressLevel;
    };

    struct Config {
        size_t sampleRate;
        size_t numberOfChannels;
        uint64_t bitRate { 0 };
        BitrateMode bitRateMode { BitrateMode::Variable };
        std::optional<OpusConfig> opusConfig { };
        std::optional<bool> isAacADTS { };
        std::optional<FlacConfig> flacConfig { };
    };
    using ActiveConfiguration = AudioEncoderActiveConfiguration;
    struct EncodedFrame {
        Vector<uint8_t> data;
        bool isKeyFrame { false };
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration { };
    };
    struct RawFrame {
        RefPtr<PlatformRawAudioData> frame;
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration { };
    };
    using CreateResult = Expected<Ref<AudioEncoder>, String>;
    using CreatePromise = NativePromise<Ref<AudioEncoder>, String>;

    using DescriptionCallback = Function<void(ActiveConfiguration&&)>;
    using OutputCallback = Function<void(EncodedFrame&&)>;
    using CreateCallback = Function<void(CreateResult&&)>;

    static Ref<CreatePromise> create(const String&, const Config&, DescriptionCallback&&, OutputCallback&&);

    using EncodePromise = NativePromise<void, String>;
    virtual Ref<EncodePromise> encode(RawFrame&&) = 0;

    virtual Ref<GenericPromise> flush() = 0;
    virtual void reset() = 0;
    virtual void close() = 0;
};

}

#endif // ENABLE(WEB_CODECS)
