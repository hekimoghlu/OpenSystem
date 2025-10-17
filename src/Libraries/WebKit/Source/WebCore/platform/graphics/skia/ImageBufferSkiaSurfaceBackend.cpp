/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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
#include "ImageBufferSkiaSurfaceBackend.h"

#if USE(SKIA)

namespace WebCore {

IntSize ImageBufferSkiaSurfaceBackend::calculateSafeBackendSize(const Parameters& parameters)
{
    IntSize backendSize = parameters.backendSize;
    if (backendSize.isEmpty())
        return backendSize;

    auto bytesPerRow = 4 * CheckedUint32(backendSize.width());
    if (bytesPerRow.hasOverflowed())
        return { };

    CheckedSize numBytes = CheckedUint32(backendSize.height()) * bytesPerRow;
    if (numBytes.hasOverflowed())
        return { };

    return backendSize;
}

unsigned ImageBufferSkiaSurfaceBackend::calculateBytesPerRow(const IntSize& backendSize)
{
    ASSERT(!backendSize.isEmpty());
    return CheckedUint32(backendSize.width()) * 4;
}

size_t ImageBufferSkiaSurfaceBackend::calculateMemoryCost(const Parameters& parameters)
{
    return ImageBufferBackend::calculateMemoryCost(parameters.backendSize, calculateBytesPerRow(parameters.backendSize));
}

ImageBufferSkiaSurfaceBackend::ImageBufferSkiaSurfaceBackend(const Parameters& parameters, sk_sp<SkSurface>&& surface, RenderingMode renderingMode)
    : ImageBufferSkiaBackend(parameters)
    , m_surface(WTFMove(surface))
    , m_context(*m_surface->getCanvas(), renderingMode, parameters.purpose)
{
    m_context.applyDeviceScaleFactor(parameters.resolutionScale);
}

ImageBufferSkiaSurfaceBackend::~ImageBufferSkiaSurfaceBackend() = default;

unsigned ImageBufferSkiaSurfaceBackend::bytesPerRow() const
{
    return m_surface->imageInfo().minRowBytes64();
}

bool ImageBufferSkiaSurfaceBackend::canMapBackingStore() const
{
    return true;
}

} // namespace WebCore

#endif // USE(SKIA)
