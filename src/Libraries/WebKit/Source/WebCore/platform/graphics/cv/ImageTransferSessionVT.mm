/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
#import "config.h"
#import "ImageTransferSessionVT.h"

#import "CVUtilities.h"
#import "GraphicsContextCG.h"
#import "Logging.h"
#import "VideoFrameCV.h"
#import <CoreMedia/CMFormatDescription.h>
#import <CoreMedia/CMSampleBuffer.h>
#import <pal/avfoundation/MediaTimeAVFoundation.h>

#if !PLATFORM(MACCATALYST)
#import <wtf/spi/cocoa/IOSurfaceSPI.h>
#endif

#import "CoreVideoSoftLink.h"
#import "VideoToolboxSoftLink.h"
#import <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

ImageTransferSessionVT::ImageTransferSessionVT(uint32_t pixelFormat, bool shouldUseIOSurface)
    : m_shouldUseIOSurface(shouldUseIOSurface)
{
    VTPixelTransferSessionRef transferSession;
    VTPixelTransferSessionCreate(kCFAllocatorDefault, &transferSession);
    ASSERT(transferSession);
    m_transferSession = adoptCF(transferSession);

    auto status = VTSessionSetProperty(transferSession, kVTPixelTransferPropertyKey_ScalingMode, kVTScalingMode_Trim);
    if (status != kCVReturnSuccess)
        RELEASE_LOG(Media, "ImageTransferSessionVT::ImageTransferSessionVT: VTSessionSetProperty(kVTPixelTransferPropertyKey_ScalingMode) failed with error %d", static_cast<int>(status));

    status = VTSessionSetProperty(transferSession, kVTPixelTransferPropertyKey_EnableHighSpeedTransfer, @YES);
    if (status != kCVReturnSuccess)
        RELEASE_LOG(Media, "ImageTransferSessionVT::ImageTransferSessionVT: VTSessionSetProperty(kVTPixelTransferPropertyKey_EnableHighSpeedTransfer) failed with error %d", static_cast<int>(status));

    status = VTSessionSetProperty(transferSession, kVTPixelTransferPropertyKey_RealTime, @YES);
    if (status != kCVReturnSuccess)
        RELEASE_LOG(Media, "ImageTransferSessionVT::ImageTransferSessionVT: VTSessionSetProperty(kVTPixelTransferPropertyKey_RealTime) failed with error %d", static_cast<int>(status));

#if PLATFORM(IOS_FAMILY) && !PLATFORM(MACCATALYST)
    status = VTSessionSetProperty(transferSession, kVTPixelTransferPropertyKey_EnableHardwareAcceleratedTransfer, @YES);
    if (status != kCVReturnSuccess)
        RELEASE_LOG(Media, "ImageTransferSessionVT::ImageTransferSessionVT: VTSessionSetProperty(kVTPixelTransferPropertyKey_EnableHardwareAcceleratedTransfer) failed with error %d", static_cast<int>(status));
#endif

    m_pixelFormat = pixelFormat;
}

void ImageTransferSessionVT::setCroppingRectangle(std::optional<FloatRect> rectangle)
{
    if (m_croppingRectangle == rectangle)
        return;

    m_croppingRectangle = rectangle;

    if (!m_croppingRectangle) {
        m_sourceCroppingDictionary = { };
        return;
    }

    m_sourceCroppingDictionary = @{
        (__bridge NSString *)kCVImageBufferCleanApertureWidthKey: @(rectangle->width()),
        (__bridge NSString *)kCVImageBufferCleanApertureHeightKey: @(rectangle->height()),
        (__bridge NSString *)kCVImageBufferCleanApertureVerticalOffsetKey: @(rectangle->x()),
        (__bridge NSString *)kCVImageBufferCleanApertureHorizontalOffsetKey: @(rectangle->y()),
    };
}

bool ImageTransferSessionVT::setSize(const IntSize& size)
{
    if (m_size == size && m_outputBufferPool)
        return true;
    auto bufferPool = createCVPixelBufferPool(size.width(), size.height(), m_pixelFormat, 6, false, m_shouldUseIOSurface);
    if (!bufferPool)
        return false;
    m_outputBufferPool = WTFMove(*bufferPool);
    m_size = size;
    return true;
}

RetainPtr<CVPixelBufferRef> ImageTransferSessionVT::convertPixelBuffer(CVPixelBufferRef sourceBuffer, const IntSize& size)
{
    if (!m_sourceCroppingDictionary && sourceBuffer && m_size == IntSize(CVPixelBufferGetWidth(sourceBuffer), CVPixelBufferGetHeight(sourceBuffer)) && m_pixelFormat == CVPixelBufferGetPixelFormatType(sourceBuffer))
        return retainPtr(sourceBuffer);

    if (m_sourceCroppingDictionary)
        CVBufferSetAttachment(sourceBuffer, kCVImageBufferCleanApertureKey, m_sourceCroppingDictionary.get(), kCVAttachmentMode_ShouldPropagate);

    if (!sourceBuffer || !setSize(size))
        return nullptr;

    auto result = createCVPixelBufferFromPool(m_outputBufferPool.get(), m_maxBufferPoolSize);
    if (!result) {
        RELEASE_LOG(Media, "ImageTransferSessionVT::convertPixelBuffer, createCVPixelBufferFromPool failed with error %d", static_cast<int>(result.error()));
        return nullptr;
    }
    auto outputBuffer = WTFMove(*result);

    auto err = VTPixelTransferSessionTransferImage(m_transferSession.get(), sourceBuffer, outputBuffer.get());
    if (err) {
        RELEASE_LOG(Media, "ImageTransferSessionVT::convertPixelBuffer, VTPixelTransferSessionTransferImage failed with error %d", static_cast<int>(err));
        return nullptr;
    }

    if (m_sourceCroppingDictionary)
        CVBufferRemoveAttachment(sourceBuffer, kCVImageBufferCleanApertureKey);

    return outputBuffer;
}

RetainPtr<CMSampleBufferRef> ImageTransferSessionVT::convertCMSampleBuffer(CMSampleBufferRef sourceBuffer, const IntSize& size, const MediaTime* sampleTime)
{
    if (!sourceBuffer)
        return nullptr;

    auto description = PAL::CMSampleBufferGetFormatDescription(sourceBuffer);
    auto sourceSize = FloatSize(PAL::CMVideoFormatDescriptionGetPresentationDimensions(description, true, true));
    auto pixelBuffer = static_cast<CVPixelBufferRef>(PAL::CMSampleBufferGetImageBuffer(sourceBuffer));
    if (size == expandedIntSize(sourceSize) && m_pixelFormat == CVPixelBufferGetPixelFormatType(pixelBuffer))
        return retainPtr(sourceBuffer);

    if (!setSize(size))
        return nullptr;

    auto convertedPixelBuffer = convertPixelBuffer(pixelBuffer, size);
    if (!convertedPixelBuffer)
        return nullptr;

    CMItemCount itemCount = 0;
    auto status = PAL::CMSampleBufferGetSampleTimingInfoArray(sourceBuffer, 1, nullptr, &itemCount);
    if (status != noErr) {
        RELEASE_LOG(Media, "ImageTransferSessionVT::convertCMSampleBuffer: CMSampleBufferGetSampleTimingInfoArray failed with error code: %d", static_cast<int>(status));
        return nullptr;
    }
    Vector<CMSampleTimingInfo> timingInfoArray;
    CMSampleTimingInfo* timeingInfoPtr = nullptr;
    if (itemCount) {
        timingInfoArray.grow(itemCount);
        status = PAL::CMSampleBufferGetSampleTimingInfoArray(sourceBuffer, itemCount, timingInfoArray.data(), nullptr);
        if (status != noErr) {
            RELEASE_LOG(Media, "ImageTransferSessionVT::convertCMSampleBuffer: CMSampleBufferGetSampleTimingInfoArray failed with error code: %d", static_cast<int>(status));
            return nullptr;
        }

        if (sampleTime) {
            auto cmTime = PAL::toCMTime(*sampleTime);
            for (auto& timing : timingInfoArray) {
                timing.presentationTimeStamp = cmTime;
                timing.decodeTimeStamp = cmTime;
            }
        }
        timeingInfoPtr = timingInfoArray.data();
    }

    CMVideoFormatDescriptionRef formatDescription = nullptr;
    status = PAL::CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, convertedPixelBuffer.get(), &formatDescription);
    if (status != noErr) {
        RELEASE_LOG(Media, "ImageTransferSessionVT::convertCMSampleBuffer: CMVideoFormatDescriptionCreateForImageBuffer returned: %d", static_cast<int>(status));
        return nullptr;
    }

    CMSampleBufferRef resizedSampleBuffer;
    status = PAL::CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault, convertedPixelBuffer.get(), formatDescription, timeingInfoPtr, &resizedSampleBuffer);
    CFRelease(formatDescription);
    if (status != noErr) {
        RELEASE_LOG(Media, "ImageTransferSessionVT::convertCMSampleBuffer: failed to create CMSampleBuffer with error code: %d", static_cast<int>(status));
        return nullptr;
    }

    return adoptCF(resizedSampleBuffer);
}

RetainPtr<CVPixelBufferRef> ImageTransferSessionVT::createPixelBuffer(CGImageRef image, const IntSize& size)
{
    if (!image || !setSize(size))
        return nullptr;

    CVPixelBufferRef rgbBuffer;
    auto imageSize = IntSize(CGImageGetWidth(image), CGImageGetHeight(image));
    auto status = CVPixelBufferCreate(kCFAllocatorDefault, imageSize.width(), imageSize.height(), kCVPixelFormatType_32ARGB, nullptr, &rgbBuffer);
    if (status != kCVReturnSuccess) {
        RELEASE_LOG(Media, "ImageTransferSessionVT::createPixelBuffer: CVPixelBufferCreate failed with error code: %d", static_cast<int>(status));
        return nullptr;
    }

    CVPixelBufferLockBaseAddress(rgbBuffer, 0);
    void* data = CVPixelBufferGetBaseAddress(rgbBuffer);
    auto retainedRGBBuffer = adoptCF(rgbBuffer);
    auto context = adoptCF(CGBitmapContextCreate(data, imageSize.width(), imageSize.height(), 8, CVPixelBufferGetBytesPerRow(rgbBuffer), sRGBColorSpaceRef(), (CGBitmapInfo) kCGImageAlphaNoneSkipFirst));
    if (!context) {
        RELEASE_LOG(Media, "ImageTransferSessionVT::createPixelBuffer: CGBitmapContextCreate returned nullptr");
        return nullptr;
    }

    CGContextDrawImage(context.get(), CGRectMake(0, 0, imageSize.width(), imageSize.height()), image);
    CVPixelBufferUnlockBaseAddress(rgbBuffer, 0);

    return convertPixelBuffer(rgbBuffer, size);
}

RetainPtr<CMSampleBufferRef> ImageTransferSessionVT::createCMSampleBuffer(CVPixelBufferRef sourceBuffer, const MediaTime& sampleTime, const IntSize& size)
{
    if (!sourceBuffer || !setSize(size))
        return nullptr;

    auto bufferSize = IntSize(CVPixelBufferGetWidth(sourceBuffer), CVPixelBufferGetHeight(sourceBuffer));
    RetainPtr<CVPixelBufferRef> inputBuffer = sourceBuffer;
    if (bufferSize != m_size || m_pixelFormat != CVPixelBufferGetPixelFormatType(sourceBuffer)) {
        inputBuffer = convertPixelBuffer(sourceBuffer, m_size);
        if (!inputBuffer)
            return nullptr;
    }

    CMVideoFormatDescriptionRef formatDescription = nullptr;
    auto status = PAL::CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, (CVImageBufferRef)inputBuffer.get(), &formatDescription);
    if (status) {
        RELEASE_LOG(Media, "ImageTransferSessionVT::convertPixelBuffer: failed to initialize CMVideoFormatDescription with error code: %d", static_cast<int>(status));
        return nullptr;
    }

    CMSampleBufferRef sampleBuffer;
    auto cmTime = PAL::toCMTime(sampleTime);
    CMSampleTimingInfo timingInfo = { PAL::kCMTimeInvalid, cmTime, cmTime };
    status = PAL::CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault, (CVImageBufferRef)inputBuffer.get(), formatDescription, &timingInfo, &sampleBuffer);
    CFRelease(formatDescription);
    if (status) {
        RELEASE_LOG(Media, "ImageTransferSessionVT::convertPixelBuffer: failed to initialize CMSampleBuffer with error code: %d", static_cast<int>(status));
        return nullptr;
    }

    return adoptCF(sampleBuffer);
}

RetainPtr<CMSampleBufferRef> ImageTransferSessionVT::createCMSampleBuffer(CGImageRef image, const MediaTime& sampleTime, const IntSize& size)
{
    auto pixelBuffer = createPixelBuffer(image, size);
    if (!pixelBuffer)
        return nullptr;

    return createCMSampleBuffer(pixelBuffer.get(), sampleTime, size);
}

#if !PLATFORM(MACCATALYST)

RetainPtr<CMSampleBufferRef> ImageTransferSessionVT::createCMSampleBuffer(IOSurfaceRef surface, const MediaTime &sampleTime, const IntSize &size)
{
    if (!surface || !setSize(size))
        return nullptr;
    auto pixelBuffer = createCVPixelBuffer(surface).value_or(nullptr);
    if (!pixelBuffer)
        return nullptr;

    return createCMSampleBuffer(pixelBuffer.get(), sampleTime, size);
}
#endif

RefPtr<VideoFrame> ImageTransferSessionVT::convertVideoFrame(VideoFrame& videoFrame, const IntSize& size)
{
    if (size == expandedIntSize(videoFrame.presentationSize()))
        return &videoFrame;

    auto resizedBuffer = convertPixelBuffer(videoFrame.pixelBuffer(), size);
    if (!resizedBuffer)
        return nullptr;

    return VideoFrameCV::create(videoFrame.presentationTime(), videoFrame.isMirrored(), videoFrame.rotation(), WTFMove(resizedBuffer));
}

RefPtr<VideoFrame> ImageTransferSessionVT::createVideoFrame(CGImageRef image, const WTF::MediaTime& time, const IntSize& size)
{
    return createVideoFrame(image, time, size, VideoFrame::Rotation::None);
}
RefPtr<VideoFrame> ImageTransferSessionVT::createVideoFrame(CMSampleBufferRef image, const WTF::MediaTime& time, const IntSize& size)
{
    return createVideoFrame(image, time, size, VideoFrame::Rotation::None);
}

#if !PLATFORM(MACCATALYST)
RefPtr<VideoFrame> ImageTransferSessionVT::createVideoFrame(IOSurfaceRef surface, const MediaTime& sampleTime, const IntSize& size)
{
    return createVideoFrame(surface, sampleTime, size, VideoFrame::Rotation::None);
}

RefPtr<VideoFrame> ImageTransferSessionVT::createVideoFrame(IOSurfaceRef surface, const MediaTime& sampleTime, const IntSize& size, VideoFrame::Rotation rotation, bool mirrored)
{
    auto sampleBuffer = createCMSampleBuffer(surface, sampleTime, size);
    if (!sampleBuffer)
        return nullptr;

    return VideoFrameCV::create(sampleBuffer.get(), mirrored, rotation);
}
#endif

RefPtr<VideoFrame> ImageTransferSessionVT::createVideoFrame(CGImageRef image, const MediaTime& sampleTime, const IntSize& size, VideoFrame::Rotation rotation, bool mirrored)
{
    auto sampleBuffer = createCMSampleBuffer(image, sampleTime, size);
    if (!sampleBuffer)
        return nullptr;

    return VideoFrameCV::create(sampleBuffer.get(), mirrored, rotation);
}

RefPtr<VideoFrame> ImageTransferSessionVT::createVideoFrame(CMSampleBufferRef buffer, const MediaTime& sampleTime, const IntSize& size, VideoFrame::Rotation rotation, bool mirrored)
{
    auto sampleBuffer = convertCMSampleBuffer(buffer, size, &sampleTime);
    if (!sampleBuffer)
        return nullptr;

    return VideoFrameCV::create(sampleBuffer.get(), mirrored, rotation);
}

} // namespace WebCore
