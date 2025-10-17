/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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

#include "WebKitUtilities.h"
#include "api/video_codecs/video_encoder.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "api/video/encoded_image.h"
#include "modules/include/module_common_types.h"
#include "modules/video_coding/include/video_error_codes.h"

using CVPixelBufferRef = struct __CVBuffer*;

namespace webrtc {

class EncodedImageCallback;
class VideoCodec;
class VideoFrame;
struct SdpVideoFormat;
class Settings;

std::unique_ptr<webrtc::VideoEncoderFactory> createWebKitEncoderFactory(WebKitH265, WebKitVP9, WebKitAv1);

using WebKitVideoEncoder = void*;
using VideoEncoderCreateCallback = WebKitVideoEncoder(*)(const SdpVideoFormat& format);
using VideoEncoderReleaseCallback = int32_t(*)(WebKitVideoEncoder);
using VideoEncoderInitializeCallback = int32_t(*)(WebKitVideoEncoder, const VideoCodec&);
using VideoEncoderEncodeCallback = int32_t(*)(WebKitVideoEncoder, const VideoFrame&, bool shouldEncodeAsKeyFrame);
using VideoEncoderRegisterEncodeCompleteCallback = int32_t(*)(WebKitVideoEncoder, void* encodedImageCallback);
using VideoEncoderSetRatesCallback = void(*)(WebKitVideoEncoder, const VideoEncoder::RateControlParameters&);

void setVideoEncoderCallbacks(VideoEncoderCreateCallback, VideoEncoderReleaseCallback, VideoEncoderInitializeCallback, VideoEncoderEncodeCallback, VideoEncoderRegisterEncodeCompleteCallback, VideoEncoderSetRatesCallback);

using WebKitEncodedFrameTiming = EncodedImage::Timing;

enum class WebKitEncodedVideoRotation : uint8_t {
    kVideoRotation_0,
    kVideoRotation_90,
    kVideoRotation_180,
    kVideoRotation_270
};

struct WebKitEncodedFrameInfo {
    uint32_t width { 0 };
    uint32_t height { 0 };
    int64_t timeStamp { 0 };
    std::optional<uint64_t> duration;
    int64_t ntpTimeMS { 0 };
    int64_t captureTimeMS { 0 };
    VideoFrameType frameType { VideoFrameType::kVideoFrameDelta };
    WebKitEncodedVideoRotation rotation { WebKitEncodedVideoRotation::kVideoRotation_0 };
    VideoContentType contentType { VideoContentType::UNSPECIFIED };
    bool completeFrame = false;
    int qp { -1 };
    int temporalIndex { -1 };
    WebKitEncodedFrameTiming timing;
};

enum class LocalEncoderScalabilityMode : uint8_t { L1T1, L1T2 };

using LocalEncoder = void*;
using LocalEncoderCallback = void (^)(const uint8_t* buffer, size_t size, const webrtc::WebKitEncodedFrameInfo&);
using LocalEncoderDescriptionCallback = void (^)(const uint8_t* buffer, size_t size);
using LocalEncoderErrorCallback = void (^)(bool isFrameDropped);
void* createLocalEncoder(const webrtc::SdpVideoFormat&, bool useAnnexB, LocalEncoderScalabilityMode, LocalEncoderCallback, LocalEncoderDescriptionCallback, LocalEncoderErrorCallback);
void releaseLocalEncoder(LocalEncoder);
void initializeLocalEncoder(LocalEncoder, uint16_t width, uint16_t height, unsigned int startBitrate, unsigned int maxBitrate, unsigned int minBitrate, uint32_t maxFramerate);
void encodeLocalEncoderFrame(LocalEncoder, CVPixelBufferRef, int64_t timeStampNs, int64_t timeStamp, std::optional<uint64_t> duration, webrtc::VideoRotation, bool isKeyframeRequired);
void setLocalEncoderRates(LocalEncoder, uint32_t bitRate, uint32_t frameRate);
void setLocalEncoderLowLatency(LocalEncoder, bool isLowLatencyEnabled);
void encoderVideoTaskComplete(void*, webrtc::VideoCodecType, const uint8_t* buffer, size_t length, const WebKitEncodedFrameInfo&);
void flushLocalEncoder(LocalEncoder);

}
