/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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

#include "api/video_codecs/video_decoder.h"
#include "VTVideoDecoderSPI.h"
#include "WebKitUtilities.h"

namespace webrtc {

class WebKitDecoderReceiver final : public DecodedImageCallback {
public:
    explicit WebKitDecoderReceiver(VTVideoDecoderSession);
    ~WebKitDecoderReceiver();

    VTVideoDecoderFrame currentFrame() const { return m_currentFrame; }
    void setCurrentFrame(VTVideoDecoderFrame currentFrame) { m_currentFrame = currentFrame; }
    OSStatus decoderFailed(int error);

    void initializeFromFormatDescription(CMFormatDescriptionRef);

private:
    int32_t Decoded(VideoFrame&) final;
    int32_t Decoded(VideoFrame&, int64_t decode_time_ms) final;
    void Decoded(VideoFrame&, std::optional<int32_t> decode_time_ms, std::optional<uint8_t> qp) final;

    CVPixelBufferPoolRef pixelBufferPool(size_t pixelBufferWidth, size_t pixelBufferHeight, BufferType);

    VTVideoDecoderSession m_session { nullptr };
    VTVideoDecoderFrame m_currentFrame { nullptr };
    size_t m_pixelBufferWidth { 0 };
    size_t m_pixelBufferHeight { 0 };
    BufferType m_bufferType { BufferType::I420 };
    bool m_isFullRange { false };
    CVPixelBufferPoolRef m_pixelBufferPool { nullptr };
};

}
