/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#include "ImageBufferShareableMappedIOSurfaceBackend.h"

#if ENABLE(GPU_PROCESS) && HAVE(IOSURFACE)

#include "Logging.h"
#include <WebCore/GraphicsContextCG.h>
#include <WebCore/IOSurfacePool.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/spi/cocoa/IOSurfaceSPI.h>
#include <wtf/text/TextStream.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageBufferShareableMappedIOSurfaceBackend);

std::unique_ptr<ImageBufferShareableMappedIOSurfaceBackend> ImageBufferShareableMappedIOSurfaceBackend::create(const Parameters& parameters, const ImageBufferCreationContext& creationContext)
{
    IntSize backendSize = calculateSafeBackendSize(parameters);
    if (backendSize.isEmpty())
        return nullptr;

    auto surface = IOSurface::create(creationContext.surfacePool, backendSize, parameters.colorSpace, IOSurface::nameForRenderingPurpose(parameters.purpose), convertToIOSurfaceFormat(parameters.pixelFormat));
    if (!surface)
        return nullptr;
    if (creationContext.resourceOwner)
        surface->setOwnershipIdentity(creationContext.resourceOwner);

    RetainPtr<CGContextRef> cgContext = surface->createPlatformContext();
    if (!cgContext)
        return nullptr;

    CGContextClearRect(cgContext.get(), FloatRect(FloatPoint::zero(), backendSize));

    return std::unique_ptr<ImageBufferShareableMappedIOSurfaceBackend> { new ImageBufferShareableMappedIOSurfaceBackend { parameters, WTFMove(surface), WTFMove(cgContext), 0, creationContext.surfacePool } };
}

std::unique_ptr<ImageBufferShareableMappedIOSurfaceBackend> ImageBufferShareableMappedIOSurfaceBackend::create(const Parameters& parameters, ImageBufferBackendHandle handle)
{
    if (!std::holds_alternative<MachSendRight>(handle)) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }

    auto surface = IOSurface::createFromSendRight(WTFMove(std::get<MachSendRight>(handle)));
    if (!surface)
        return nullptr;
    auto cgContext = surface->createPlatformContext();
    if (!cgContext)
        return nullptr;

    ASSERT(surface->colorSpace() == parameters.colorSpace);
    return std::unique_ptr<ImageBufferShareableMappedIOSurfaceBackend> { new ImageBufferShareableMappedIOSurfaceBackend { parameters, WTFMove(surface), WTFMove(cgContext), 0, nullptr } };
}

std::optional<ImageBufferBackendHandle> ImageBufferShareableMappedIOSurfaceBackend::createBackendHandle(SharedMemory::Protection) const
{
    return ImageBufferBackendHandle(m_surface->createSendRight());
}

String ImageBufferShareableMappedIOSurfaceBackend::debugDescription() const
{
    TextStream stream;
    stream << "ImageBufferShareableMappedIOSurfaceBackend " << this << " " << ValueOrNull(m_surface.get());
    return stream.release();
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && HAVE(IOSURFACE)
