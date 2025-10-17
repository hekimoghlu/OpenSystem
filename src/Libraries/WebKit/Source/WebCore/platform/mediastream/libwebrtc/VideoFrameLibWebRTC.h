/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 17, 2023.
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

#if PLATFORM(COCOA) && USE(LIBWEBRTC)

#include "VideoFrame.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/api/video/video_frame.h>
#include <webrtc/webkit_sdk/WebKit/WebKitUtilities.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

using CVPixelBufferRef = struct __CVBuffer*;


namespace WebCore {

class VideoFrameLibWebRTC final : public VideoFrame {
public:
    using ConversionCallback = std::function<RetainPtr<CVPixelBufferRef>(webrtc::VideoFrameBuffer&)>;
    static RefPtr<VideoFrameLibWebRTC> create(MediaTime, bool isMirrored, Rotation, std::optional<PlatformVideoColorSpace>&&, rtc::scoped_refptr<webrtc::VideoFrameBuffer>&&, ConversionCallback&&);

    rtc::scoped_refptr<webrtc::VideoFrameBuffer> buffer() const { return m_buffer; }

    static std::optional<PlatformVideoColorSpace> colorSpaceFromFrame(const webrtc::VideoFrame&);

private:
    VideoFrameLibWebRTC(MediaTime, bool isMirrored, Rotation, PlatformVideoColorSpace&&, rtc::scoped_refptr<webrtc::VideoFrameBuffer>&&, ConversionCallback&&);

    // VideoFrame
    IntSize presentationSize() const final { return m_size; }
    uint32_t pixelFormat() const final { return m_videoPixelFormat; }
    CVPixelBufferRef pixelBuffer() const final;

    Ref<VideoFrame> clone() final;

    const rtc::scoped_refptr<webrtc::VideoFrameBuffer> m_buffer;
    IntSize m_size;
    uint32_t m_videoPixelFormat { 0 };

    mutable ConversionCallback m_conversionCallback;
    mutable RetainPtr<CVPixelBufferRef> m_pixelBuffer WTF_GUARDED_BY_LOCK(m_pixelBufferLock);
    mutable Lock m_pixelBufferLock;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::VideoFrameLibWebRTC)
    static bool isType(const WebCore::VideoFrame& videoFrame) { return videoFrame.isLibWebRTC(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // PLATFORM(COCOA) && USE(LIBWEBRTC)
