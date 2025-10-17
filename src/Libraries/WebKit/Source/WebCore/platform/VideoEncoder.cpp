/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#include "config.h"
#include "VideoEncoder.h"

#if ENABLE(VIDEO)

#if USE(LIBWEBRTC) && PLATFORM(COCOA)
#include "LibWebRTCVPXVideoEncoder.h"
#endif

#if USE(GSTREAMER)
#include "VideoEncoderGStreamer.h"
#endif

namespace WebCore {

VideoEncoder::CreatorFunction VideoEncoder::s_customCreator = nullptr;

void VideoEncoder::setCreatorCallback(CreatorFunction&& function)
{
    s_customCreator = WTFMove(function);
}

Ref<VideoEncoder::CreatePromise> VideoEncoder::create(const String& codecName, const Config& config, DescriptionCallback&& descriptionCallback, OutputCallback&& outputCallback)
{
    CreatePromise::Producer producer;
    Ref promise = producer.promise();
    CreateCallback callback = [producer = WTFMove(producer)] (auto&& result) mutable {
        producer.settle(WTFMove(result));
    };

    if (s_customCreator) {
        s_customCreator(codecName, config, WTFMove(callback), WTFMove(descriptionCallback), WTFMove(outputCallback));
        return promise;
    }
    createLocalEncoder(codecName, config, WTFMove(callback), WTFMove(descriptionCallback), WTFMove(outputCallback));
    return promise;
}

void VideoEncoder::createLocalEncoder(const String& codecName, const Config& config, CreateCallback&& callback, DescriptionCallback&& descriptionCallback, OutputCallback&& outputCallback)
{
#if USE(LIBWEBRTC) && PLATFORM(COCOA)
    if (codecName == "vp8"_s) {
        LibWebRTCVPXVideoEncoder::create(LibWebRTCVPXVideoEncoder::Type::VP8, config, WTFMove(callback), WTFMove(descriptionCallback), WTFMove(outputCallback));
        return;
    }
    if (codecName.startsWith("vp09.00"_s)) {
        LibWebRTCVPXVideoEncoder::create(LibWebRTCVPXVideoEncoder::Type::VP9, config, WTFMove(callback), WTFMove(descriptionCallback), WTFMove(outputCallback));
        return;
    }
    if (codecName.startsWith("vp09.02"_s)) {
        LibWebRTCVPXVideoEncoder::create(LibWebRTCVPXVideoEncoder::Type::VP9_P2, config, WTFMove(callback), WTFMove(descriptionCallback), WTFMove(outputCallback));
        return;
    }
#if ENABLE(AV1)
    if (codecName.startsWith("av01."_s)) {
        LibWebRTCVPXVideoEncoder::create(LibWebRTCVPXVideoEncoder::Type::AV1, config, WTFMove(callback), WTFMove(descriptionCallback), WTFMove(outputCallback));
        return;
    }
#endif
#elif USE(GSTREAMER)
    GStreamerVideoEncoder::create(codecName, config, WTFMove(callback), WTFMove(descriptionCallback), WTFMove(outputCallback));
    return;
#else
    UNUSED_PARAM(codecName);
    UNUSED_PARAM(config);
    UNUSED_PARAM(descriptionCallback);
    UNUSED_PARAM(outputCallback);
#endif

    callback(makeUnexpected("Not supported"_s));
}

}

#endif // ENABLE(VIDEO)
