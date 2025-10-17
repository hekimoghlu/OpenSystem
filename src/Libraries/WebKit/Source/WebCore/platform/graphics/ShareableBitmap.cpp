/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
#include "ShareableBitmap.h"

#include "GraphicsContext.h"
#include "SharedMemory.h"
#include <wtf/DebugHeap.h>

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(ShareableBitmap, WTF_INTERNAL);
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ShareableBitmap);

ShareableBitmapConfiguration::ShareableBitmapConfiguration(const IntSize& size, std::optional<DestinationColorSpace> colorSpace, bool isOpaque)
    : m_size(size)
    , m_colorSpace(validateColorSpace(colorSpace))
    , m_isOpaque(isOpaque)
    , m_bytesPerPixel(calculateBytesPerPixel(this->colorSpace()))
    , m_bytesPerRow(calculateBytesPerRow(size, this->colorSpace()))
#if USE(CG)
    , m_bitmapInfo(calculateBitmapInfo(this->colorSpace(), isOpaque))
#endif
#if USE(SKIA)
    , m_imageInfo(SkImageInfo::MakeN32Premul(size.width(), size.height(), this->colorSpace().platformColorSpace()))
#endif
{
    ASSERT(!m_size.isEmpty());
}

ShareableBitmapConfiguration::ShareableBitmapConfiguration(const IntSize& size, std::optional<DestinationColorSpace> colorSpace, bool isOpaque, unsigned bytesPerPixel, unsigned bytesPerRow
#if USE(CG)
    , CGBitmapInfo bitmapInfo
#endif
)
    : m_size(size)
    , m_colorSpace(colorSpace)
    , m_isOpaque(isOpaque)
    , m_bytesPerPixel(bytesPerPixel)
    , m_bytesPerRow(bytesPerRow)
#if USE(CG)
    , m_bitmapInfo(bitmapInfo)
#endif
#if USE(SKIA)
    , m_imageInfo(SkImageInfo::MakeN32Premul(size.width(), size.height(), this->colorSpace().platformColorSpace()))
#endif
{
    // This constructor is called when decoding ShareableBitmapConfiguration. So this constructor
    // will behave like the default constructor if a null ShareableBitmapHandle was encoded.
}

CheckedUint32 ShareableBitmapConfiguration::calculateSizeInBytes(const IntSize& size, const DestinationColorSpace& colorSpace)
{
    return calculateBytesPerRow(size, colorSpace) * size.height();
}

RefPtr<ShareableBitmap> ShareableBitmap::create(const ShareableBitmapConfiguration& configuration)
{
    auto sizeInBytes = configuration.sizeInBytes();
    if (sizeInBytes.hasOverflowed())
        return nullptr;

    RefPtr<SharedMemory> sharedMemory = SharedMemory::allocate(sizeInBytes);
    if (!sharedMemory)
        return nullptr;

    return adoptRef(new ShareableBitmap(configuration, sharedMemory.releaseNonNull()));
}

RefPtr<ShareableBitmap> ShareableBitmap::create(const ShareableBitmapConfiguration& configuration, Ref<SharedMemory>&& sharedMemory)
{
    auto sizeInBytes = configuration.sizeInBytes();
    if (sizeInBytes.hasOverflowed())
        return nullptr;

    if (sharedMemory->size() < sizeInBytes) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }

    return adoptRef(new ShareableBitmap(configuration, WTFMove(sharedMemory)));
}

RefPtr<ShareableBitmap> ShareableBitmap::createFromImageDraw(NativeImage& image, const DestinationColorSpace& colorSpace)
{
    return createFromImageDraw(image, colorSpace, image.size());
}

RefPtr<ShareableBitmap> ShareableBitmap::createFromImageDraw(NativeImage& image, const DestinationColorSpace& colorSpace, const IntSize& destinationSize)
{
    return createFromImageDraw(image, colorSpace, destinationSize, destinationSize);
}

RefPtr<ShareableBitmap> ShareableBitmap::createFromImageDraw(NativeImage& image, const DestinationColorSpace& colorSpace, const IntSize& destinationSize, const IntSize& sourceSize)
{
    auto bitmap = ShareableBitmap::create({ destinationSize, colorSpace });
    if (!bitmap)
        return nullptr;

    auto context = bitmap->createGraphicsContext();
    if (!context)
        return nullptr;

    context->drawNativeImage(image, FloatRect({ }, destinationSize), FloatRect({ }, sourceSize), { CompositeOperator::Copy });
    return bitmap;
}

RefPtr<ShareableBitmap> ShareableBitmap::create(Handle&& handle, SharedMemory::Protection protection)
{
    auto sharedMemory = SharedMemory::map(WTFMove(handle.m_handle), protection);
    if (!sharedMemory)
        return nullptr;

    return create(handle.m_configuration, sharedMemory.releaseNonNull());
}

std::optional<Ref<ShareableBitmap>> ShareableBitmap::createReadOnly(std::optional<Handle>&& handle)
{
    if (!handle)
        return std::nullopt;

    auto sharedMemory = SharedMemory::map(WTFMove(handle->m_handle), SharedMemory::Protection::ReadOnly);
    if (!sharedMemory)
        return std::nullopt;

    return adoptRef(*new ShareableBitmap(handle->m_configuration, sharedMemory.releaseNonNull()));
}

auto ShareableBitmap::createHandle(SharedMemory::Protection protection) const -> std::optional<Handle>
{
    auto memoryHandle = m_sharedMemory->createHandle(protection);
    if (!memoryHandle)
        return std::nullopt;
    return { Handle(WTFMove(*memoryHandle), m_configuration) };
}

auto ShareableBitmap::createReadOnlyHandle() const -> std::optional<Handle>
{
    return createHandle(SharedMemory::Protection::ReadOnly);
}

ShareableBitmap::ShareableBitmap(ShareableBitmapConfiguration configuration, Ref<SharedMemory>&& sharedMemory)
    : m_configuration(configuration)
    , m_sharedMemory(WTFMove(sharedMemory))
{
}

std::span<const uint8_t> ShareableBitmap::span() const
{
    return m_sharedMemory->span();
}

std::span<uint8_t> ShareableBitmap::mutableSpan()
{
    return m_sharedMemory->mutableSpan();
}

} // namespace WebCore
