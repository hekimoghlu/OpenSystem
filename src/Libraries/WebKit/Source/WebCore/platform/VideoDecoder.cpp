/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#include "VideoDecoder.h"

#if USE(LIBWEBRTC) && PLATFORM(COCOA)
#include "LibWebRTCVPXVideoDecoder.h"
#include "WebRTCProvider.h"
#endif

#if USE(GSTREAMER)
#include "VideoDecoderGStreamer.h"
#endif

#include <wtf/UniqueRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

VideoDecoder::CreatorFunction VideoDecoder::s_customCreator = nullptr;

void VideoDecoder::setCreatorCallback(CreatorFunction&& function)
{
    s_customCreator = WTFMove(function);
}

bool VideoDecoder::isVPXSupported()
{
#if USE(LIBWEBRTC) && PLATFORM(COCOA)
    return WebRTCProvider::webRTCAvailable();
#elif USE(GSTREAMER)
    return true;
#else
    return false;
#endif
}

Ref<VideoDecoder::CreatePromise> VideoDecoder::create(const String& codecName, const Config& config, OutputCallback&& outputCallback)
{
    CreatePromise::Producer producer;
    Ref promise = producer.promise();
    CreateCallback callback = [producer = WTFMove(producer)] (auto&& result) mutable {
        producer.settle(WTFMove(result));
    };

    if (s_customCreator) {
        s_customCreator(codecName, config, WTFMove(callback), WTFMove(outputCallback));
        return promise;
    }
    createLocalDecoder(codecName, config, WTFMove(callback), WTFMove(outputCallback));
    return promise;
}

#define LE_CHR(a, b, c, d) (((a)<<24) | ((b)<<16) | ((c)<<8) | (d))

String VideoDecoder::fourCCToCodecString(uint32_t fourCC)
{
    switch (fourCC) {
    case LE_CHR('v', 'p', '0', '8'): return "vp8"_s;
    case LE_CHR('v', 'p', '0', '9'): return "vp09.00"_s;
    case LE_CHR('a', 'v', '0', '1'): return "av01."_s;
    default:
            return nullString();
    }
}

void VideoDecoder::createLocalDecoder(const String& codecName, const Config& config, CreateCallback&& callback, OutputCallback&& outputCallback)
{
#if USE(LIBWEBRTC) && PLATFORM(COCOA)
    if (codecName == "vp8"_s) {
        LibWebRTCVPXVideoDecoder::create(LibWebRTCVPXVideoDecoder::Type::VP8, config, WTFMove(callback), WTFMove(outputCallback));
        return;
    }
    if (codecName.startsWith("vp09.00"_s)) {
        LibWebRTCVPXVideoDecoder::create(LibWebRTCVPXVideoDecoder::Type::VP9, config, WTFMove(callback), WTFMove(outputCallback));
        return;
    }
    if (codecName.startsWith("vp09.02"_s)) {
        LibWebRTCVPXVideoDecoder::create(LibWebRTCVPXVideoDecoder::Type::VP9_P2, config, WTFMove(callback), WTFMove(outputCallback));
        return;
    }
#if ENABLE(AV1)
    if (codecName.startsWith("av01."_s)) {
        LibWebRTCVPXVideoDecoder::create(LibWebRTCVPXVideoDecoder::Type::AV1, config, WTFMove(callback), WTFMove(outputCallback));
        return;
    }
#endif
#elif USE(GSTREAMER)
    GStreamerVideoDecoder::create(codecName, config, WTFMove(callback), WTFMove(outputCallback));
    return;
#else
    UNUSED_PARAM(codecName);
    UNUSED_PARAM(config);
    UNUSED_PARAM(outputCallback);
#endif

    callback(makeUnexpected("Not supported"_s));
}

VideoDecoder::VideoDecoder() = default;
VideoDecoder::~VideoDecoder() = default;

}
