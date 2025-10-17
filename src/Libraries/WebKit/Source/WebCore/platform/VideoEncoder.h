/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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

#include "VideoEncoderActiveConfiguration.h"
#include "VideoEncoderScalabilityMode.h"
#include "VideoFrame.h"
#include <span>
#include <wtf/CompletionHandler.h>
#include <wtf/NativePromise.h>

namespace WebCore {

class VideoEncoder : public ThreadSafeRefCounted<VideoEncoder> {
public:
    virtual ~VideoEncoder() = default;

    using ScalabilityMode = VideoEncoderScalabilityMode;

    struct Config {
        uint64_t width { 0 };
        uint64_t height { 0 };
        bool useAnnexB { false };
        uint64_t bitRate { 0 };
        double frameRate { 0 };
        bool isRealtime { true };
        ScalabilityMode scalabilityMode { ScalabilityMode::L1T1 };
    };
    using ActiveConfiguration = VideoEncoderActiveConfiguration;
    struct EncodedFrame {
        Vector<uint8_t> data;
        bool isKeyFrame { false };
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration;
        std::optional<unsigned> temporalIndex;
    };
    struct RawFrame {
        Ref<VideoFrame> frame;
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration;
    };
    using CreateResult = Expected<Ref<VideoEncoder>, String>;
    using CreatePromise = NativePromise<Ref<VideoEncoder>, String>;

    using DescriptionCallback = Function<void(ActiveConfiguration&&)>;
    using OutputCallback = Function<void(EncodedFrame&&)>;
    using CreateCallback = Function<void(CreateResult&&)>;

    using CreatorFunction = void(*)(const String&, const Config&, CreateCallback&&, DescriptionCallback&&, OutputCallback&&);
    WEBCORE_EXPORT static void setCreatorCallback(CreatorFunction&&);

    static Ref<CreatePromise> create(const String&, const Config&, DescriptionCallback&&, OutputCallback&&);
    WEBCORE_EXPORT static void createLocalEncoder(const String&, const Config&, CreateCallback&&, DescriptionCallback&&, OutputCallback&&);

    using EncodePromise = NativePromise<void, String>;
    virtual Ref<EncodePromise> encode(RawFrame&&, bool shouldGenerateKeyFrame) = 0;

    virtual Ref<GenericPromise> setRates(uint64_t /* bitRate */, double /* frameRate */) = 0;
    virtual Ref<GenericPromise> flush() = 0;
    virtual void reset() = 0;
    virtual void close() = 0;

    static CreatorFunction s_customCreator;
};

}

#endif // ENABLE(VIDEO)
