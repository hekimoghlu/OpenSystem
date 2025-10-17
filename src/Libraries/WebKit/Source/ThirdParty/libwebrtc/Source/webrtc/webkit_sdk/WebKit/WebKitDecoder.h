/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

#include <Availability.h>
#include <TargetConditionals.h>
#include "WebKitUtilities.h"
#include "api/video/encoded_image.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "rtc_base/ref_counted_object.h"

typedef struct __CVBuffer* CVPixelBufferRef;

namespace webrtc {

#if (TARGET_OS_OSX || TARGET_OS_MACCATALYST) && TARGET_CPU_X86_64
    #define CMBASE_OBJECT_NEEDS_ALIGNMENT 1
#else
    #define CMBASE_OBJECT_NEEDS_ALIGNMENT 0
#endif

struct SdpVideoFormat;
class VideoDecoderFactory;

struct WebKitVideoDecoder {
    using Value = void*;
    Value value { nullptr };
    bool isWebRTCVideoDecoder { false };
};
using VideoDecoderCreateCallback = WebKitVideoDecoder(*)(const SdpVideoFormat& format);
using VideoDecoderReleaseCallback = int32_t(*)(WebKitVideoDecoder::Value);
using VideoDecoderDecodeCallback = int32_t(*)(WebKitVideoDecoder::Value, uint32_t timeStamp, const uint8_t*, size_t length, uint16_t width, uint16_t height);
using VideoDecoderRegisterDecodeCompleteCallback = int32_t(*)(WebKitVideoDecoder::Value, void* decodedImageCallback);

void setVideoDecoderCallbacks(VideoDecoderCreateCallback, VideoDecoderReleaseCallback, VideoDecoderDecodeCallback, VideoDecoderRegisterDecodeCompleteCallback);

std::unique_ptr<webrtc::VideoDecoderFactory> createWebKitDecoderFactory(WebKitH265, WebKitVP9, WebKitVP9VTB, WebKitAv1);
void videoDecoderTaskComplete(void* callback, uint32_t timeStampRTP, CVPixelBufferRef);
void videoDecoderTaskComplete(void* callback, uint32_t timeStampRTP, void*, GetBufferCallback, ReleaseBufferCallback, int width, int height);

using LocalDecoder = void*;
using LocalDecoderCallback = void (^)(CVPixelBufferRef, int64_t timeStamp, int64_t timeStampNs, bool isReordered);
void* createLocalH264Decoder(LocalDecoderCallback);
void* createLocalH265Decoder(LocalDecoderCallback);
void* createLocalVP9Decoder(LocalDecoderCallback);
void releaseLocalDecoder(LocalDecoder);
void flushLocalDecoder(LocalDecoder);
int32_t setDecodingFormat(LocalDecoder, const uint8_t*, size_t, uint16_t width, uint16_t height);
int32_t decodeFrame(LocalDecoder, int64_t timeStamp, const uint8_t*, size_t);
void setDecoderFrameSize(LocalDecoder, uint16_t width, uint16_t height);

class WebKitEncodedImageBufferWrapper : public EncodedImageBufferInterface {
public:
    static rtc::scoped_refptr<WebKitEncodedImageBufferWrapper> create(uint8_t* data, size_t size) { return rtc::make_ref_counted<WebKitEncodedImageBufferWrapper>(data, size); }

    WebKitEncodedImageBufferWrapper(uint8_t* data, size_t size)
        : m_data(data)
        , m_size(size)
    {
    }

    const uint8_t* data() const final { return m_data; }
    uint8_t* data() final { return m_data; }
    size_t size() const final { return m_size; }

 private:
    uint8_t* m_data { nullptr };
    size_t m_size { 0 };
};

}
