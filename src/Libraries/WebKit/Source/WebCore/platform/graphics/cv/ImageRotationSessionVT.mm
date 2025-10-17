/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
#import "ImageRotationSessionVT.h"

#import "AffineTransform.h"
#import "CVUtilities.h"
#import "Logging.h"
#import "VideoFrame.h"

#import "CoreVideoSoftLink.h"
#import "VideoToolboxSoftLink.h"
#import <pal/cf/CoreMediaSoftLink.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageRotationSessionVT);

static ImageRotationSessionVT::RotationProperties transformToRotationProperties(const AffineTransform& inTransform)
{
    ImageRotationSessionVT::RotationProperties rotation;
    if (inTransform.isIdentity())
        return rotation;

    AffineTransform::DecomposedType decomposed { };
    if (!inTransform.decompose(decomposed))
        return rotation;

    rotation.flipY = WTF::areEssentiallyEqual(decomposed.scaleX, -1.);
    rotation.flipX = WTF::areEssentiallyEqual(decomposed.scaleY, -1.);
    auto degrees = rad2deg(decomposed.angle);
    while (degrees < 0)
        degrees += 360;

    // Only support rotation in multiples of 90Âº:
    if (WTF::areEssentiallyEqual(fmod(degrees, 90.), 0.))
        rotation.angle = clampToUnsigned(degrees);

    return rotation;
}

ImageRotationSessionVT::ImageRotationSessionVT(AffineTransform&& transform, FloatSize size, IsCGImageCompatible isCGImageCompatible, ShouldUseIOSurface shouldUseIOSurface)
    : ImageRotationSessionVT(transformToRotationProperties(transform), size, isCGImageCompatible, shouldUseIOSurface)
{
    m_transform = WTFMove(transform);
}

ImageRotationSessionVT::ImageRotationSessionVT(const RotationProperties& rotation, FloatSize size, IsCGImageCompatible isCGImageCompatible, ShouldUseIOSurface shouldUseIOSurface)
    : m_shouldUseIOSurface(shouldUseIOSurface == ShouldUseIOSurface::Yes)
{
    initialize(rotation, size, isCGImageCompatible);
}

void ImageRotationSessionVT::initialize(const RotationProperties& rotation, FloatSize size, IsCGImageCompatible isCGImageCompatible)
{
    m_rotationProperties = rotation;
    m_size = size;
    m_isCGImageCompatible = isCGImageCompatible;

    if (m_rotationProperties.angle == 90 || m_rotationProperties.angle == 270)
        size = size.transposedSize();

    m_rotatedSize = expandedIntSize(size);
    m_rotationPool = nullptr;

    VTImageRotationSessionRef rawRotationSession = nullptr;
    VTImageRotationSessionCreate(kCFAllocatorDefault, m_rotationProperties.angle, &rawRotationSession);
    m_rotationSession = adoptCF(rawRotationSession);
    VTImageRotationSessionSetProperty(m_rotationSession.get(), kVTImageRotationPropertyKey_EnableHighSpeedTransfer, kCFBooleanTrue);

    if (m_rotationProperties.flipY)
        VTImageRotationSessionSetProperty(m_rotationSession.get(), kVTImageRotationPropertyKey_FlipVerticalOrientation, kCFBooleanTrue);
    if (m_rotationProperties.flipX)
        VTImageRotationSessionSetProperty(m_rotationSession.get(), kVTImageRotationPropertyKey_FlipHorizontalOrientation, kCFBooleanTrue);
}

RetainPtr<CVPixelBufferRef> ImageRotationSessionVT::rotate(CVPixelBufferRef pixelBuffer)
{
    auto pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    if (pixelFormat != m_pixelFormat || !m_rotationPool) {
        m_pixelFormat = pixelFormat;
        auto bufferPool = createCVPixelBufferPool(m_rotatedSize.width(), m_rotatedSize.height(), m_pixelFormat, 0u, m_isCGImageCompatible == IsCGImageCompatible::Yes, m_shouldUseIOSurface);

        if (!bufferPool) {
            RELEASE_LOG_ERROR(WebRTC, "ImageRotationSessionVT failed creating buffer pool with error %d", (int)bufferPool.error());
            return nullptr;
        }

        m_rotationPool = WTFMove(*bufferPool);
    }

    RetainPtr<CVPixelBufferRef> result;
    CVPixelBufferRef rawRotatedBuffer = nullptr;
    auto status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, m_rotationPool.get(), &rawRotatedBuffer);
    if (status != kCVReturnSuccess || !rawRotatedBuffer) {
        RELEASE_LOG_ERROR(WebRTC, "ImageRotationSessionVT failed creating buffer from pool with error %d", status);
        return nullptr;
    }
    result = adoptCF(rawRotatedBuffer);

    status = VTImageRotationSessionTransferImage(m_rotationSession.get(), pixelBuffer, rawRotatedBuffer);
    if (status != noErr) {
        RELEASE_LOG_ERROR(WebRTC, "ImageRotationSessionVT failed rotating buffer with error %d", status);
        return nullptr;
    }

    return result;
}

RetainPtr<CVPixelBufferRef> ImageRotationSessionVT::rotate(VideoFrame& videoFrame, const RotationProperties& rotation, IsCGImageCompatible cgImageCompatible)
{
    auto pixelBuffer = videoFrame.pixelBuffer();
    ASSERT(pixelBuffer);
    if (!pixelBuffer)
        return nullptr;

    m_pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    IntSize size { (int)CVPixelBufferGetWidth(pixelBuffer), (int)CVPixelBufferGetHeight(pixelBuffer) };
    if (rotation != m_rotationProperties || m_size != size)
        initialize(rotation, size, cgImageCompatible);

    return rotate(pixelBuffer);
}

}
