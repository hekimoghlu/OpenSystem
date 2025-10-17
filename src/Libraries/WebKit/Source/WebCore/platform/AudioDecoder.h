/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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

#include <span>
#include <wtf/CompletionHandler.h>
#include <wtf/NativePromise.h>

namespace WebCore {

class PlatformRawAudioData;

class AudioDecoder : public ThreadSafeRefCounted<AudioDecoder> {
public:
    WEBCORE_EXPORT AudioDecoder();
    WEBCORE_EXPORT virtual ~AudioDecoder();

    static bool isCodecSupported(const StringView&);

    struct Config {
        Vector<uint8_t> description;
        uint64_t sampleRate { 0 };
        uint64_t numberOfChannels { 0 };
    };

    struct EncodedData {
        std::span<const uint8_t> data;
        bool isKeyFrame { false };
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration;
    };
    struct DecodedData {
        Ref<PlatformRawAudioData> data;
    };

    using OutputCallback = Function<void(Expected<DecodedData, String>&&)>;
    using CreateResult = Expected<Ref<AudioDecoder>, String>;
    using CreatePromise = NativePromise<Ref<AudioDecoder>, String>;
    using CreateCallback = Function<void(CreateResult&&)>;

    using CreatorFunction = void(*)(const String&, const Config&, CreateCallback&&, OutputCallback&&);
    WEBCORE_EXPORT static void setCreatorCallback(CreatorFunction&&);

    static Ref<CreatePromise> create(const String&, const Config&, OutputCallback&&);

    using DecodePromise = NativePromise<void, String>;
    virtual Ref<DecodePromise> decode(EncodedData&&) = 0;

    virtual Ref<GenericPromise> flush() = 0;
    virtual void reset() = 0;
    virtual void close() = 0;
};

}

#endif // ENABLE(WEB_CODECS)
