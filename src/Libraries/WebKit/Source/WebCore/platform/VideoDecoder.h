/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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

#include "PlatformVideoColorSpace.h"
#include "ProcessIdentity.h"
#include <span>
#include <wtf/CompletionHandler.h>
#include <wtf/NativePromise.h>

namespace WebCore {

class VideoFrame;

class VideoDecoder : public ThreadSafeRefCounted<VideoDecoder> {
public:
    WEBCORE_EXPORT virtual ~VideoDecoder();

    enum class HardwareAcceleration : bool { No, Yes };
    enum class HardwareBuffer : bool { No, Yes };
    enum class TreatNoOutputAsError : bool { No, Yes };
    struct Config {
        Vector<uint8_t> description;
        uint64_t width { 0 };
        uint64_t height { 0 };
        std::optional<PlatformVideoColorSpace> colorSpace;
        HardwareAcceleration decoding { HardwareAcceleration::No };
        HardwareBuffer pixelBuffer { HardwareBuffer::No };
        TreatNoOutputAsError noOutputAsError { TreatNoOutputAsError::Yes };
        ProcessIdentity resourceOwner { };
    };

    struct EncodedFrame {
        std::span<const uint8_t> data;
        bool isKeyFrame { false };
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration;
    };
    struct DecodedFrame {
        Ref<VideoFrame> frame;
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration;
    };

    static bool isVPXSupported();

    using OutputCallback = Function<void(Expected<DecodedFrame, String>&&)>;
    using CreateResult = Expected<Ref<VideoDecoder>, String>;
    using CreatePromise = NativePromise<Ref<VideoDecoder>, String>;
    using CreateCallback = Function<void(CreateResult&&)>;

    using CreatorFunction = void(*)(const String&, const Config&, CreateCallback&&, OutputCallback&&);
    WEBCORE_EXPORT static void setCreatorCallback(CreatorFunction&&);

    static Ref<CreatePromise> create(const String&, const Config&, OutputCallback&&);
    WEBCORE_EXPORT static void createLocalDecoder(const String&, const Config&, CreateCallback&&, OutputCallback&&);

    using DecodePromise = NativePromise<void, String>;
    virtual Ref<DecodePromise> decode(EncodedFrame&&) = 0;

    virtual Ref<GenericPromise> flush() = 0;
    virtual void reset() = 0;
    virtual void close() = 0;

    static String fourCCToCodecString(uint32_t fourCC);

    static CreatorFunction s_customCreator;
protected:
    WEBCORE_EXPORT VideoDecoder();

};

}
