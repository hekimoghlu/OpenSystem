/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#include "WebKitVP9Decoder.h"

#include "WebKitDecoderReceiver.h"
#include "modules/video_coding/include/video_error_codes.h"
#include "rtc_base/logging.h"

namespace webrtc {

WebKitDecoderReceiver::WebKitDecoderReceiver(VTVideoDecoderSession session)
    : m_session(session)
{
}

WebKitDecoderReceiver::~WebKitDecoderReceiver()
{
    if (m_pixelBufferPool)
        CFRelease(m_pixelBufferPool);
}

void WebKitDecoderReceiver::initializeFromFormatDescription(CMFormatDescriptionRef formatDescription)
{
    // CoreAnimation doesn't support full-planar YUV, so we must convert the buffers output
    // by libvpx to bi-planar YUV. Create pixel buffer attributes and give those to the
    // decoder session for use in creating its own internal CVPixelBufferPool, which we
    // will use post-decode.
    m_isFullRange = false;
    m_bufferType = BufferType::I420;

    auto extensions = CMFormatDescriptionGetExtensions(formatDescription);
    if (!extensions)
        return;

    CFTypeRef extensionAtoms = CFDictionaryGetValue(extensions, kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms);
    if (!extensionAtoms || CFGetTypeID(extensionAtoms) != CFDictionaryGetTypeID())
        return;

    auto configurationRecord = static_cast<CFDataRef>(CFDictionaryGetValue((CFDictionaryRef)extensionAtoms, CFSTR("vpcC")));
    if (!configurationRecord || CFGetTypeID(configurationRecord) != CFDataGetTypeID())
        return;

    auto configurationRecordSize = CFDataGetLength(configurationRecord);
    if (configurationRecordSize < 12)
        return;

    auto configurationRecordData = CFDataGetBytePtr(configurationRecord);
    auto bitDepthChromaAndRange = *(configurationRecordData + 6);

    m_isFullRange = bitDepthChromaAndRange & 0x1;
    uint8_t bitDepth = bitDepthChromaAndRange >> 4;
    uint8_t chromaSubsampling = (bitDepthChromaAndRange & 0xf0) >> 1;
    if (!chromaSubsampling || chromaSubsampling == 1)
        m_bufferType = bitDepth == 10 ? BufferType::I010 : BufferType::I420;
    else if (chromaSubsampling == 2)
        m_bufferType = bitDepth == 10 ? BufferType::I210 : BufferType::I422;
}

CVPixelBufferPoolRef WebKitDecoderReceiver::pixelBufferPool(size_t pixelBufferWidth, size_t pixelBufferHeight, BufferType type)
{
    if (m_pixelBufferPool && m_pixelBufferWidth == pixelBufferWidth && m_pixelBufferHeight == pixelBufferHeight && m_bufferType == type)
        return m_pixelBufferPool;

    OSType pixelFormat;
    if (type == BufferType::I420 || type == BufferType::I422)
        pixelFormat = m_isFullRange ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
    else if (type == BufferType::I010)
        pixelFormat = m_isFullRange ? kCVPixelFormatType_420YpCbCr10BiPlanarFullRange : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange;
    else if (type == BufferType::I210)
        pixelFormat = m_isFullRange ? kCVPixelFormatType_422YpCbCr10BiPlanarFullRange : kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange;
    else
        return nil;

    auto createPixelFormatAttributes = [] (OSType pixelFormat) {
        auto createNumber = [] (int32_t format) -> CFNumberRef {
            return CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &format);
        };
        auto cfPixelFormats = CFArrayCreateMutable(kCFAllocatorDefault, 2, &kCFTypeArrayCallBacks);
        auto formatNumber = createNumber(pixelFormat);
        CFArrayAppendValue(cfPixelFormats, formatNumber);
        CFRelease(formatNumber);

        auto borderPixelsValue = createNumber(32);

        const void* keys[] = {
            kCVPixelBufferPixelFormatTypeKey,
            kCVPixelBufferExtendedPixelsLeftKey,
            kCVPixelBufferExtendedPixelsRightKey,
            kCVPixelBufferExtendedPixelsTopKey,
            kCVPixelBufferExtendedPixelsBottomKey,
        };
        const void* values[] = {
            cfPixelFormats,
            borderPixelsValue,
            borderPixelsValue,
            borderPixelsValue,
            borderPixelsValue,
        };
        auto attributes = CFDictionaryCreate(kCFAllocatorDefault, keys, values, std::size(keys), &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        CFRelease(borderPixelsValue);
        CFRelease(cfPixelFormats);
        return attributes;
    };

    auto pixelBufferAttributes = createPixelFormatAttributes(pixelFormat);
    VTDecoderSessionSetPixelBufferAttributes(m_session, pixelBufferAttributes);
    CFRelease(pixelBufferAttributes);

    if (m_pixelBufferPool) {
        CFRelease(m_pixelBufferPool);
        m_pixelBufferPool = nullptr;
    }

    m_pixelBufferPool = VTDecoderSessionGetPixelBufferPool(m_session);
    if (m_pixelBufferPool)
        CFRetain(m_pixelBufferPool);

    m_pixelBufferWidth = pixelBufferWidth;
    m_pixelBufferHeight = pixelBufferHeight;
    m_bufferType = type;

    return m_pixelBufferPool;
}

OSStatus WebKitDecoderReceiver::decoderFailed(int error)
{
    OSStatus vtError;
    if (error == WEBRTC_VIDEO_CODEC_NO_OUTPUT)
        vtError = noErr;
    else if (error == WEBRTC_VIDEO_CODEC_UNINITIALIZED)
        vtError = kVTVideoDecoderMalfunctionErr;
    else if (error == WEBRTC_VIDEO_CODEC_MEMORY)
        vtError = kVTAllocationFailedErr;
    else
        vtError = kVTVideoDecoderBadDataErr;

    VTDecoderSessionEmitDecodedFrame(m_session, m_currentFrame, vtError, vtError ? 0 : kVTDecodeInfo_FrameDropped, nullptr);
    m_currentFrame = nullptr;

    RTC_LOG(LS_ERROR) << "VP9 decoder: decoder failed with error " << error << ", vtError " << vtError;
    return vtError;
}

int32_t WebKitDecoderReceiver::Decoded(VideoFrame& frame)
{
    auto pixelBuffer = createPixelBufferFromFrame(frame, [this](size_t width, size_t height, BufferType type) -> CVPixelBufferRef {
        auto pixelBufferPool = this->pixelBufferPool(width, height, type);
        if (!pixelBufferPool)
            return nullptr;

        CVPixelBufferRef pixelBuffer = nullptr;
        if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, m_pixelBufferPool, &pixelBuffer) == kCVReturnSuccess)
            return pixelBuffer;
        return nullptr;
    });

    VTDecoderSessionEmitDecodedFrame(m_session, m_currentFrame, pixelBuffer ? noErr : -1, 0, pixelBuffer);
    m_currentFrame = nullptr;
    if (pixelBuffer)
        CFRelease(pixelBuffer);
    return 0;
}

int32_t WebKitDecoderReceiver::Decoded(VideoFrame& frame, int64_t)
{
    Decoded(frame);
    return 0;
}

void WebKitDecoderReceiver::Decoded(VideoFrame& frame, std::optional<int32_t>, std::optional<uint8_t>)
{
    Decoded(frame);
}

}
