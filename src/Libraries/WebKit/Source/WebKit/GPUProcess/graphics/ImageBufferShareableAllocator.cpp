/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include "ImageBufferShareableAllocator.h"

#include "ImageBufferShareableBitmapBackend.h"
#include "ShareablePixelBuffer.h"
#include <WebCore/GraphicsContext.h>
#include <WebCore/ImageBuffer.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(GPU_PROCESS)

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageBufferShareableAllocator);

ImageBufferShareableAllocator::ImageBufferShareableAllocator(const ProcessIdentity& resourceOwner)
    : m_resourceOwner(resourceOwner)
{
}

RefPtr<ImageBuffer> ImageBufferShareableAllocator::createImageBuffer(const FloatSize& size, const DestinationColorSpace& colorSpace, RenderingMode) const
{
    RefPtr<ImageBuffer> imageBuffer = ImageBuffer::create<ImageBufferShareableBitmapBackend>(size, 1, colorSpace, ImageBufferPixelFormat::BGRA8, RenderingPurpose::Unspecified, { });
    if (!imageBuffer)
        return nullptr;

    auto* sharing = imageBuffer->toBackendSharing();
    ASSERT(is<ImageBufferBackendHandleSharing>(sharing));

    auto bitmap = downcast<ImageBufferBackendHandleSharing>(*sharing).bitmap();
    if (!bitmap)
        return nullptr;

    auto handle = bitmap->createHandle();
    if (!handle)
        return nullptr;

    transferMemoryOwnership(WTFMove(handle->handle()));
    return imageBuffer;
}

RefPtr<PixelBuffer> ImageBufferShareableAllocator::createPixelBuffer(const PixelBufferFormat& format, const IntSize& size) const
{
    RefPtr pixelBuffer = ShareablePixelBuffer::tryCreate(format, size);
    if (!pixelBuffer)
        return nullptr;

    auto handle = pixelBuffer->protectedData()->createHandle(SharedMemory::Protection::ReadOnly);
    if (!handle)
        return nullptr;

    transferMemoryOwnership(WTFMove(*handle));
    return pixelBuffer;
}

void ImageBufferShareableAllocator::transferMemoryOwnership(SharedMemory::Handle&& handle) const
{
    handle.setOwnershipOfMemory(m_resourceOwner, MemoryLedger::Graphics);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
