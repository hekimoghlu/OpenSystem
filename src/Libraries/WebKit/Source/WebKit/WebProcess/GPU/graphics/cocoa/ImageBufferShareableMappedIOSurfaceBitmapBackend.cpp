/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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
#include "ImageBufferShareableMappedIOSurfaceBitmapBackend.h"

#if ENABLE(GPU_PROCESS) && HAVE(IOSURFACE)

#include "Logging.h"
#include <WebCore/GraphicsContextCG.h>
#include <WebCore/IOSurfacePool.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/spi/cocoa/IOSurfaceSPI.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageBufferShareableMappedIOSurfaceBitmapBackend);

std::unique_ptr<ImageBufferShareableMappedIOSurfaceBitmapBackend> ImageBufferShareableMappedIOSurfaceBitmapBackend::create(const Parameters& parameters, const ImageBufferCreationContext& creationContext)
{
    IntSize backendSize = ImageBufferIOSurfaceBackend::calculateSafeBackendSize(parameters);
    if (backendSize.isEmpty())
        return nullptr;

    auto surface = IOSurface::create(creationContext.surfacePool, backendSize, parameters.colorSpace, IOSurface::Name::ImageBuffer, convertToIOSurfaceFormat(parameters.pixelFormat));
    if (!surface)
        return nullptr;
    if (creationContext.resourceOwner)
        surface->setOwnershipIdentity(creationContext.resourceOwner);
    auto lockAndContext = surface->createBitmapPlatformContext();
    if (!lockAndContext)
        return nullptr;
    CGContextClearRect(lockAndContext->context.get(), FloatRect(FloatPoint::zero(), backendSize));
    return makeUnique<ImageBufferShareableMappedIOSurfaceBitmapBackend>(parameters, WTFMove(surface), WTFMove(*lockAndContext), creationContext.surfacePool);
}

ImageBufferShareableMappedIOSurfaceBitmapBackend::ImageBufferShareableMappedIOSurfaceBitmapBackend(const Parameters& parameters, std::unique_ptr<IOSurface> surface, IOSurface::LockAndContext&& lockAndContext, IOSurfacePool* ioSurfacePool)
    : ImageBufferCGBackend(parameters)
    , m_surface(WTFMove(surface))
    , m_lock(WTFMove(lockAndContext.lock))
    , m_ioSurfacePool(ioSurfacePool)
{
    m_context = makeUnique<GraphicsContextCG>(lockAndContext.context.get());
    applyBaseTransform(*m_context);
}

ImageBufferShareableMappedIOSurfaceBitmapBackend::~ImageBufferShareableMappedIOSurfaceBitmapBackend()
{
    releaseGraphicsContext();
    IOSurface::moveToPool(WTFMove(m_surface), m_ioSurfacePool.get());
}

bool ImageBufferShareableMappedIOSurfaceBitmapBackend::canMapBackingStore() const
{
    return true;
}

std::optional<ImageBufferBackendHandle> ImageBufferShareableMappedIOSurfaceBitmapBackend::createBackendHandle(SharedMemory::Protection) const
{
    return ImageBufferBackendHandle(m_surface->createSendRight());
}

GraphicsContext& ImageBufferShareableMappedIOSurfaceBitmapBackend::context()
{
    if (m_context) {
        CGContextRef cgContext = m_context->platformContext();
        if (m_lock || !cgContext) {
            // The existing context is a valid context and the IOSurface is locked, or alternatively
            // the existing context is an invalid context, for some reason we ran into an error previously.
            return *m_context;
        }

        // The IOSurface is unlocked for every flush to prepare for external access by the compositor.
        // Re-lock on first context() request after the external access has ended and new update starts.
        if (auto lock = m_surface->lock<IOSurface::AccessMode::ReadWrite>()) {
            if (lock->surfaceBaseAddress() == CGBitmapContextGetData(cgContext)) {
                m_lock = WTFMove(lock);
                return *m_context;
            }
        }
        m_context = nullptr;
    } else {
        auto lockAndContext = m_surface->createBitmapPlatformContext();
        if (lockAndContext) {
            m_lock = WTFMove(lockAndContext->lock);
            m_context = makeUnique<GraphicsContextCG>(lockAndContext->context.get());
            applyBaseTransform(*m_context);
            return *m_context;
        }
    }
    // For some reason we ran into an error. Construct an invalid context, with current API we must
    // return an object.
    RELEASE_LOG(RemoteLayerBuffers, "ImageBufferShareableMappedIOSurfaceBitmapBackend::context() - failed to create or update the context");
    m_context = makeUnique<GraphicsContextCG>(nullptr);
    applyBaseTransform(*m_context);
    return *m_context;
}

unsigned ImageBufferShareableMappedIOSurfaceBitmapBackend::bytesPerRow() const
{
    return m_surface->bytesPerRow();
}

RefPtr<NativeImage> ImageBufferShareableMappedIOSurfaceBitmapBackend::copyNativeImage()
{
    ASSERT_NOT_REACHED(); // Not applicable for LayerBacking.
    return nullptr;
}

RefPtr<NativeImage> ImageBufferShareableMappedIOSurfaceBitmapBackend::createNativeImageReference()
{
    ASSERT_NOT_REACHED(); // Not applicable for LayerBacking.
    return nullptr;
}

RefPtr<NativeImage> ImageBufferShareableMappedIOSurfaceBitmapBackend::sinkIntoNativeImage()
{
    ASSERT_NOT_REACHED(); // Not applicable for LayerBacking.
    return nullptr;
}

bool ImageBufferShareableMappedIOSurfaceBitmapBackend::isInUse() const
{
    return m_surface->isInUse();
}

void ImageBufferShareableMappedIOSurfaceBitmapBackend::releaseGraphicsContext()
{
    m_context = nullptr;
    m_lock = std::nullopt;
}

bool ImageBufferShareableMappedIOSurfaceBitmapBackend::setVolatile()
{
    if (m_surface->isInUse())
        return false;

    setVolatilityState(VolatilityState::Volatile);
    m_surface->setVolatile(true);
    return true;
}

SetNonVolatileResult ImageBufferShareableMappedIOSurfaceBitmapBackend::setNonVolatile()
{
    setVolatilityState(VolatilityState::NonVolatile);
    return m_surface->setVolatile(false);
}

VolatilityState ImageBufferShareableMappedIOSurfaceBitmapBackend::volatilityState() const
{
    return m_volatilityState;
}

void ImageBufferShareableMappedIOSurfaceBitmapBackend::setVolatilityState(VolatilityState volatilityState)
{
    m_volatilityState = volatilityState;
}

void ImageBufferShareableMappedIOSurfaceBitmapBackend::transferToNewContext(const ImageBufferCreationContext&)
{
    ASSERT_NOT_REACHED(); // Not applicable for LayerBacking.
}

void ImageBufferShareableMappedIOSurfaceBitmapBackend::getPixelBuffer(const IntRect&, PixelBuffer&)
{
    ASSERT_NOT_REACHED(); // Not applicable for LayerBacking.
}

void ImageBufferShareableMappedIOSurfaceBitmapBackend::putPixelBuffer(const PixelBuffer&, const IntRect&, const IntPoint&, AlphaPremultiplication)
{
    ASSERT_NOT_REACHED(); // Not applicable for LayerBacking.
}

void ImageBufferShareableMappedIOSurfaceBitmapBackend::flushContext()
{
    // Flush means external access by the compositor. Unlock the IOSurface so that the compositor sees the updates to the bitmap.
    m_lock = std::nullopt;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && HAVE(IOSURFACE)
