/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#import "CVUtilities.h"

#import "ColorSpaceCG.h"
#import "IOSurface.h"
#import "Logging.h"
#import "RealtimeVideoUtilities.h"
#import <wtf/StdLibExtras.h>
#import <wtf/cf/TypeCastsCF.h>

#import "CoreVideoSoftLink.h"

namespace WebCore {

static Expected<RetainPtr<CVPixelBufferPoolRef>, CVReturn> createBufferPool(unsigned minimumBufferCount, NSDictionary *pixelAttributes)
{
    NSDictionary *poolOptions = nullptr;
    if (minimumBufferCount) {
        poolOptions = @{
            (__bridge NSString *)kCVPixelBufferPoolMinimumBufferCountKey : @(minimumBufferCount)
        };
    }
    CVPixelBufferPoolRef pool = nullptr;
    auto status = CVPixelBufferPoolCreate(kCFAllocatorDefault, (__bridge CFDictionaryRef)poolOptions, (__bridge CFDictionaryRef)pixelAttributes, &pool);
    if (status != kCVReturnSuccess || !pool)
        return makeUnexpected(status);
    return adoptCF(pool);
}

Expected<RetainPtr<CVPixelBufferPoolRef>, CVReturn> createIOSurfaceCVPixelBufferPool(size_t width, size_t height, OSType format, unsigned minimumBufferCount, bool isCGImageCompatible)
{
    return createBufferPool(minimumBufferCount, @{
        (__bridge NSString *)kCVPixelBufferWidthKey : @(width),
        (__bridge NSString *)kCVPixelBufferHeightKey : @(height),
        (__bridge NSString *)kCVPixelBufferPixelFormatTypeKey : @(format),
        (__bridge NSString *)kCVPixelBufferCGImageCompatibilityKey : isCGImageCompatible ? @YES : @NO,
#if PLATFORM(MAC)
        (__bridge NSString *)kCVPixelBufferOpenGLCompatibilityKey : @YES,
#endif
        (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{ }
    });
}

Expected<RetainPtr<CVPixelBufferPoolRef>, CVReturn> createInMemoryCVPixelBufferPool(size_t width, size_t height, OSType format, unsigned minimumBufferCount, bool isCGImageCompatible)
{
    return createBufferPool(minimumBufferCount, @{
        (__bridge NSString *)kCVPixelBufferWidthKey : @(width),
        (__bridge NSString *)kCVPixelBufferHeightKey : @(height),
        (__bridge NSString *)kCVPixelBufferPixelFormatTypeKey : @(format),
        (__bridge NSString *)kCVPixelBufferCGImageCompatibilityKey : isCGImageCompatible ? @YES : @NO,
#if PLATFORM(MAC)
        (__bridge NSString *)kCVPixelBufferOpenGLCompatibilityKey : @YES,
#endif
    });
}

Expected<RetainPtr<CVPixelBufferPoolRef>, CVReturn> createCVPixelBufferPool(size_t width, size_t height, OSType format, unsigned minimumBufferCount, bool isCGImageCompatible, bool shouldUseIOSurfacePool)
{
    return shouldUseIOSurfacePool ? createIOSurfaceCVPixelBufferPool(width, height, format, minimumBufferCount, isCGImageCompatible) : createInMemoryCVPixelBufferPool(width, height, format, minimumBufferCount, isCGImageCompatible);
}

Expected<RetainPtr<CVPixelBufferRef>, CVReturn> createCVPixelBufferFromPool(CVPixelBufferPoolRef pixelBufferPool, unsigned maxBufferSize)
{
    CVPixelBufferRef pixelBuffer = nullptr;
    CVReturn status;
    if (!maxBufferSize)
        status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pixelBufferPool, &pixelBuffer);
    else {
        auto *auxiliaryAttributes = @{ (__bridge NSString *)kCVPixelBufferPoolAllocationThresholdKey : @(maxBufferSize) };
        status = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(kCFAllocatorDefault, pixelBufferPool, (__bridge CFDictionaryRef)auxiliaryAttributes, &pixelBuffer);
    }
    if (status != kCVReturnSuccess || !pixelBuffer)
        return makeUnexpected(status);
    return adoptCF(pixelBuffer);
}

static CFDictionaryRef pixelBufferCreationOptions(IOSurfaceRef surface)
{
#if PLATFORM(MAC)
    auto format = IOSurfaceGetPixelFormat(surface);
    if (format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
        // YUV formats might contain extra pixels due to block size. Define the picture position
        // in the buffer as the default top-left position. The caller might use the pixel buffer
        // with APIs that expect the properties.
        constexpr size_t macroblockSize = 16;
        auto width = IOSurfaceGetWidth(surface);
        auto height = IOSurfaceGetHeight(surface);
        auto extendedRight = roundUpToMultipleOf(macroblockSize, width) - width;
        auto extendedBottom = roundUpToMultipleOf(macroblockSize, height) - height;
        if ((IOSurfaceGetBytesPerRowOfPlane(surface, 0) >= width + extendedRight)
            && (IOSurfaceGetBytesPerRowOfPlane(surface, 1) >= width + extendedRight)
            && (IOSurfaceGetAllocSize(surface) >= (height + extendedBottom) * IOSurfaceGetBytesPerRowOfPlane(surface, 0) * 3 / 2)) {
            return (__bridge CFDictionaryRef) @{
                (__bridge NSString *)kCVPixelBufferOpenGLCompatibilityKey : @YES,
                (__bridge NSString *)kCVPixelBufferExtendedPixelsRightKey : @(extendedRight),
                (__bridge NSString *)kCVPixelBufferExtendedPixelsBottomKey : @(extendedBottom)
            };
        }
    }
#else
    UNUSED_PARAM(surface);
#endif
    return (__bridge CFDictionaryRef) @{
#if PLATFORM(MAC)
        (__bridge NSString *)kCVPixelBufferOpenGLCompatibilityKey : @YES
#endif
    };
}

Expected<RetainPtr<CVPixelBufferRef>, CVReturn> createCVPixelBuffer(IOSurfaceRef surface)
{
    CVPixelBufferRef pixelBuffer = nullptr;
    auto status = CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, surface, pixelBufferCreationOptions(surface), &pixelBuffer);
    if (status != kCVReturnSuccess || !pixelBuffer) {
        RELEASE_LOG_ERROR(WebRTC, "createCVPixelBuffer failed with IOSurface status=%d, pixelBuffer=%p", (int)status, pixelBuffer);
        return makeUnexpected(status);
    }
    return adoptCF(pixelBuffer);
}

RetainPtr<CGColorSpaceRef> createCGColorSpaceForCVPixelBuffer(CVPixelBufferRef buffer)
{
    if (CGColorSpaceRef colorSpace = dynamic_cf_cast<CGColorSpaceRef>(CVBufferGetAttachment(buffer, kCVImageBufferCGColorSpaceKey, nullptr)))
        return colorSpace;

    RetainPtr<CFDictionaryRef> attachments;
#if HAVE(CVBUFFERCOPYATTACHMENTS)
    attachments = adoptCF(CVBufferCopyAttachments(buffer, kCVAttachmentMode_ShouldPropagate));
#else
    attachments = CVBufferGetAttachments(buffer, kCVAttachmentMode_ShouldPropagate);
#endif
    if (auto colorSpace = adoptCF(CVImageBufferCreateColorSpaceFromAttachments(attachments.get())))
        return colorSpace;

    // We should only get here with content that has a broken embedded ICC
    // profile; in all other cases VideoToolbox should have put either known
    // accurate or guessed color space attachments on the pixel buffer. Content
    // that requires an embedded ICC profile is unlikely to be presented
    // correctly with any particular fallback color space we choose, so we
    // choose sRGB for ease.
    return sRGBColorSpaceRef();
}

void setOwnershipIdentityForCVPixelBuffer(CVPixelBufferRef pixelBuffer, const ProcessIdentity& owner)
{
    auto surface = CVPixelBufferGetIOSurface(pixelBuffer);
    ASSERT(surface);
    IOSurface::setOwnershipIdentity(surface, owner);
}

RetainPtr<CVPixelBufferRef> createBlackPixelBuffer(size_t width, size_t height, bool shouldUseIOSurface)
{
    OSType format = preferedPixelBufferFormat();
    ASSERT(format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);

    NSDictionary *pixelAttributes = @{ (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{ } };

    CVPixelBufferRef pixelBuffer = nullptr;
    auto status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, shouldUseIOSurface ? (__bridge CFDictionaryRef)pixelAttributes : nullptr, &pixelBuffer);
    if (status != noErr || !pixelBuffer)
        return nullptr;

    status = CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    ASSERT(status == noErr);

    auto yPlane = CVPixelBufferGetSpanOfPlane(pixelBuffer, 0);
    size_t yStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    for (unsigned i = 0; i < height; ++i)
        zeroSpan(yPlane.subspan(i * yStride, width));

    auto uvPlane = CVPixelBufferGetSpanOfPlane(pixelBuffer, 1);
    size_t uvStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    for (unsigned i = 0; i < height / 2; ++i)
        memsetSpan(uvPlane.subspan(i * uvStride, width), 128);

    status = CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    ASSERT(!status);
    return adoptCF(pixelBuffer);
}

}
