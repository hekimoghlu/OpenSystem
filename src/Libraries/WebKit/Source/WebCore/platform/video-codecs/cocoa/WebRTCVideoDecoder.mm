/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#include "WebRTCVideoDecoder.h"
#include <wtf/TZoneMallocInlines.h>

#if USE(LIBWEBRTC)

#import "RTCVideoDecoderVTBAV1.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/webkit_sdk/WebKit/WebKitDecoder.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

class WebRTCLocalVideoDecoder final : public WebRTCVideoDecoder {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebRTCLocalVideoDecoder);
public:
    explicit WebRTCLocalVideoDecoder(webrtc::LocalDecoder decoder)
        : m_decoder(decoder)
    {
    }

    ~WebRTCLocalVideoDecoder()
    {
        webrtc::releaseLocalDecoder(m_decoder);
    }

private:
    void flush() final { webrtc::flushLocalDecoder(m_decoder); }
    void setFormat(std::span<const uint8_t> data, uint16_t width, uint16_t height) final { webrtc::setDecodingFormat(m_decoder, data.data(), data.size(), width, height); }
    int32_t decodeFrame(int64_t timeStamp, std::span<const uint8_t> data) final { return webrtc::decodeFrame(m_decoder, timeStamp, data.data(), data.size()); }
    void setFrameSize(uint16_t width, uint16_t height) final { webrtc::setDecoderFrameSize(m_decoder, width, height); }

    webrtc::LocalDecoder m_decoder;
};

class WebRTCDecoderVTBAV1 final : public WebRTCVideoDecoder {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebRTCDecoderVTBAV1);
public:
    explicit WebRTCDecoderVTBAV1(RTCVideoDecoderVTBAV1Callback callback)
        : m_decoder(adoptNS([[RTCVideoDecoderVTBAV1 alloc] init]))
    {
        [m_decoder setCallback:callback];
    }

    ~WebRTCDecoderVTBAV1()
    {
        [m_decoder releaseDecoder];
    }

private:
    void flush() final { [m_decoder flush]; }
    void setFormat(std::span<const uint8_t>, uint16_t width, uint16_t height) final { setFrameSize(width, height); }
    int32_t decodeFrame(int64_t timeStamp, std::span<const uint8_t> data) final { return [m_decoder decodeData:data.data() size:data.size() timeStamp:timeStamp]; }
    void setFrameSize(uint16_t width, uint16_t height) final { [m_decoder setWidth:width height:height];; }

    RetainPtr<RTCVideoDecoderVTBAV1> m_decoder;
};

std::unique_ptr<WebRTCVideoDecoder> WebRTCVideoDecoder::create(VideoCodecType decoderType, WebRTCVideoDecoderCallback callback)
{
    switch (decoderType) {
    case VideoCodecType::H264:
        return makeUnique<WebRTCLocalVideoDecoder>(webrtc::createLocalH264Decoder(callback));
    case VideoCodecType::H265:
        return makeUnique<WebRTCLocalVideoDecoder>(webrtc::createLocalH265Decoder(callback));
    case VideoCodecType::VP9:
        return makeUnique<WebRTCLocalVideoDecoder>(webrtc::createLocalVP9Decoder(callback));
    case VideoCodecType::AV1:
        return makeUnique<WebRTCDecoderVTBAV1>(callback);
    }
    ASSERT_NOT_REACHED();
    return nullptr;
}

}

#endif //  USE(LIBWEBRTC)
