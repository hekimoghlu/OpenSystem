/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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

#if ENABLE(VIDEO) && PLATFORM(COCOA)

#include <span>
#include <wtf/RetainPtr.h>

typedef struct __CVBuffer* CVPixelBufferRef;
typedef struct __CVPixelBufferPool* CVPixelBufferPoolRef;

namespace webrtc {
class VideoFrameBuffer;
}

namespace WebCore {

class SharedVideoFrameInfo {
public:
    SharedVideoFrameInfo() = default;
    SharedVideoFrameInfo(OSType, uint32_t width, uint32_t height, uint32_t bytesPerRow, uint32_t widthPlaneB = 0, uint32_t heightPlaneB = 0, uint32_t bytesPerRowPlaneB = 0, uint32_t bytesPerRowPlaneA = 0);

    WEBCORE_EXPORT void encode(std::span<uint8_t>);
    WEBCORE_EXPORT static std::optional<SharedVideoFrameInfo> decode(std::span<const uint8_t>);

    WEBCORE_EXPORT static SharedVideoFrameInfo fromCVPixelBuffer(CVPixelBufferRef);
    WEBCORE_EXPORT bool writePixelBuffer(CVPixelBufferRef, std::span<uint8_t> data);

#if USE(LIBWEBRTC)
    WEBCORE_EXPORT static SharedVideoFrameInfo fromVideoFrameBuffer(const webrtc::VideoFrameBuffer&);
    WEBCORE_EXPORT bool writeVideoFrameBuffer(webrtc::VideoFrameBuffer&, std::span<uint8_t> data);
#endif

    WEBCORE_EXPORT size_t storageSize() const;

    WEBCORE_EXPORT RetainPtr<CVPixelBufferRef> createPixelBufferFromMemory(std::span<const uint8_t> data, CVPixelBufferPoolRef = nullptr);

    WEBCORE_EXPORT bool isReadWriteSupported() const;
    WEBCORE_EXPORT RetainPtr<CVPixelBufferPoolRef> createCompatibleBufferPool() const;

    OSType bufferType() const { return m_bufferType; }
    uint32_t width() const { return m_width; };
    uint32_t height() const { return m_height; };

private:
    OSType m_bufferType { 0 };
    uint32_t m_width { 0 };
    uint32_t m_height { 0 };
    uint32_t m_bytesPerRow { 0 };
    uint32_t m_widthPlaneB { 0 };
    uint32_t m_heightPlaneB { 0 };
    uint32_t m_bytesPerRowPlaneB { 0 };
    uint32_t m_bytesPerRowPlaneAlpha { 0 };
    size_t m_storageSize { 0 };
};


static constexpr size_t SharedVideoFrameInfoEncodingLength = sizeof(SharedVideoFrameInfo);

inline SharedVideoFrameInfo::SharedVideoFrameInfo(OSType bufferType, uint32_t width, uint32_t height, uint32_t bytesPerRow, uint32_t widthPlaneB, uint32_t heightPlaneB, uint32_t bytesPerRowPlaneB, uint32_t bytesPerRowPlaneAlpha)
    : m_bufferType(bufferType)
    , m_width(width)
    , m_height(height)
    , m_bytesPerRow(bytesPerRow)
    , m_widthPlaneB(widthPlaneB)
    , m_heightPlaneB(heightPlaneB)
    , m_bytesPerRowPlaneB(bytesPerRowPlaneB)
    , m_bytesPerRowPlaneAlpha(bytesPerRowPlaneAlpha)
{
}

}

#endif // ENABLE(MEDIA_STREAM)
