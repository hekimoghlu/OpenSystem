/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#include "ImageBufferShareableBitmapBackend.h"

#include <WebCore/GraphicsContext.h>
#include <WebCore/PixelBuffer.h>
#include <WebCore/ShareableBitmap.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include <WebCore/GraphicsContextCG.h>
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageBufferShareableBitmapBackend);

IntSize ImageBufferShareableBitmapBackend::calculateSafeBackendSize(const Parameters& parameters)
{
    IntSize backendSize = parameters.backendSize;
    if (backendSize.isEmpty())
        return { };

    CheckedUint32 numBytes = ShareableBitmapConfiguration::calculateSizeInBytes(backendSize, parameters.colorSpace);
    if (numBytes.hasOverflowed())
        return { };

    return backendSize;
}

unsigned ImageBufferShareableBitmapBackend::calculateBytesPerRow(const Parameters& parameters, const IntSize& backendSize)
{
    ASSERT(!backendSize.isEmpty());
    return ShareableBitmapConfiguration::calculateBytesPerRow(backendSize, parameters.colorSpace);
}

size_t ImageBufferShareableBitmapBackend::calculateMemoryCost(const Parameters& parameters)
{
    return ImageBufferBackend::calculateMemoryCost(parameters.backendSize, calculateBytesPerRow(parameters, parameters.backendSize));
}

std::unique_ptr<ImageBufferShareableBitmapBackend> ImageBufferShareableBitmapBackend::create(const Parameters& parameters, const ImageBufferCreationContext& creationContext)
{
    ASSERT(parameters.pixelFormat == ImageBufferPixelFormat::BGRA8 || parameters.pixelFormat == ImageBufferPixelFormat::BGRX8);

    IntSize backendSize = calculateSafeBackendSize(parameters);
    if (backendSize.isEmpty())
        return nullptr;

    auto bitmap = ShareableBitmap::create({ backendSize, parameters.colorSpace });
    if (!bitmap)
        return nullptr;
    if (creationContext.resourceOwner)
        bitmap->setOwnershipOfMemory(creationContext.resourceOwner);
    auto context = bitmap->createGraphicsContext();
    if (!context)
        return nullptr;

    return makeUnique<ImageBufferShareableBitmapBackend>(parameters, bitmap.releaseNonNull(), WTFMove(context));
}

std::unique_ptr<ImageBufferShareableBitmapBackend> ImageBufferShareableBitmapBackend::create(const Parameters& parameters, ShareableBitmap::Handle handle)
{
    auto bitmap = ShareableBitmap::create(WTFMove(handle));
    if (!bitmap)
        return nullptr;

    auto context = bitmap->createGraphicsContext();
    if (!context)
        return nullptr;

    return makeUnique<ImageBufferShareableBitmapBackend>(parameters, bitmap.releaseNonNull(), WTFMove(context));
}

ImageBufferShareableBitmapBackend::ImageBufferShareableBitmapBackend(const Parameters& parameters, Ref<ShareableBitmap>&& bitmap, std::unique_ptr<GraphicsContext>&& context)
    : ImageBufferShareableBitmapBackendBase(parameters)
    , m_bitmap(WTFMove(bitmap))
    , m_context(WTFMove(context))
{
    // ShareableBitmap ensures that the coordinate space in the context that we're adopting
    // has a top-left origin, so we don't ever need to flip here, so we don't call setupContext().
    // However, ShareableBitmap does not have a notion of scale, so we must apply the device
    // scale factor to the context ourselves.
    m_context->applyDeviceScaleFactor(resolutionScale());
}

ImageBufferShareableBitmapBackend::~ImageBufferShareableBitmapBackend() = default;

bool ImageBufferShareableBitmapBackend::canMapBackingStore() const
{
    return true;
}

std::optional<ImageBufferBackendHandle> ImageBufferShareableBitmapBackend::createBackendHandle(SharedMemory::Protection protection) const
{
    if (auto handle = m_bitmap->createHandle(protection))
        return ImageBufferBackendHandle(WTFMove(*handle));
    return { };
}

void ImageBufferShareableBitmapBackend::transferToNewContext(const ImageBufferCreationContext& creationContext)
{
    if (creationContext.resourceOwner)
        m_bitmap->setOwnershipOfMemory(creationContext.resourceOwner);
}

unsigned ImageBufferShareableBitmapBackend::bytesPerRow() const
{
    return m_bitmap->bytesPerRow();
}

#if USE(CAIRO)
RefPtr<cairo_surface_t> ImageBufferShareableBitmapBackend::createCairoSurface()
{
    return m_bitmap->createPersistentCairoSurface();
}
#endif

RefPtr<NativeImage> ImageBufferShareableBitmapBackend::copyNativeImage()
{
    return NativeImage::create(m_bitmap->createPlatformImage(CopyBackingStore));
}

RefPtr<NativeImage> ImageBufferShareableBitmapBackend::createNativeImageReference()
{
    return NativeImage::create(m_bitmap->createPlatformImage(DontCopyBackingStore));
}

void ImageBufferShareableBitmapBackend::getPixelBuffer(const IntRect& srcRect, PixelBuffer& destination)
{
    ImageBufferBackend::getPixelBuffer(srcRect, m_bitmap->span(), destination);
}

void ImageBufferShareableBitmapBackend::putPixelBuffer(const PixelBuffer& pixelBuffer, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat)
{
    ImageBufferBackend::putPixelBuffer(pixelBuffer, srcRect, destPoint, destFormat, m_bitmap->mutableSpan());
}

String ImageBufferShareableBitmapBackend::debugDescription() const
{
    TextStream stream;
    stream << "ImageBufferShareableBitmapBackend " << this;
    return stream.release();
}

} // namespace WebKit
