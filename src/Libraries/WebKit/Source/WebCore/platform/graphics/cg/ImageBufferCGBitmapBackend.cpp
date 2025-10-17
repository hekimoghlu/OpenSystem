/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#include "ImageBufferCGBitmapBackend.h"

#if USE(CG)

#include "GraphicsContext.h"
#include "GraphicsContextCG.h"
#include "ImageBufferUtilitiesCG.h"
#include "IntRect.h"
#include "PixelBuffer.h"
#include <wtf/CheckedArithmetic.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/MallocSpan.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageBufferCGBitmapBackend);

size_t ImageBufferCGBitmapBackend::calculateMemoryCost(const Parameters& parameters)
{
    return ImageBufferBackend::calculateMemoryCost(parameters.backendSize, calculateBytesPerRow(parameters.backendSize));
}

std::unique_ptr<ImageBufferCGBitmapBackend> ImageBufferCGBitmapBackend::create(const Parameters& parameters, const ImageBufferCreationContext&)
{
    ASSERT(parameters.pixelFormat == ImageBufferPixelFormat::BGRA8);

    IntSize backendSize = calculateSafeBackendSize(parameters);
    if (backendSize.isEmpty())
        return nullptr;

    CheckedSize bytesPerRow = checkedProduct<size_t>(4, backendSize.width());
    if (bytesPerRow.hasOverflowed())
        return nullptr;

    CheckedSize numBytes = checkedProduct<size_t>(backendSize.height(), bytesPerRow);
    if (numBytes.hasOverflowed())
        return nullptr;

    auto data = MallocSpan<uint8_t>::tryZeroedMalloc(numBytes);
    if (!data)
        return nullptr;

    ASSERT(!(reinterpret_cast<intptr_t>(data.span().data()) & 3));

    verifyImageBufferIsBigEnough(data.span());

    RetainPtr cgContext = adoptCF(CGBitmapContextCreate(data.mutableSpan().data(), backendSize.width(), backendSize.height(), 8, bytesPerRow, parameters.colorSpace.platformColorSpace(), static_cast<uint32_t>(kCGImageAlphaPremultipliedFirst) | static_cast<uint32_t>(kCGBitmapByteOrder32Host)));
    if (!cgContext)
        return nullptr;

    auto context = makeUnique<GraphicsContextCG>(cgContext.get());

    RetainPtr dataProvider = adoptCF(CGDataProviderCreateWithData(nullptr, data.mutableSpan().data(), numBytes, [] (void*, const void* data, size_t) {
        fastFree(const_cast<void*>(data));
    }));

    return std::unique_ptr<ImageBufferCGBitmapBackend>(new ImageBufferCGBitmapBackend(parameters, data.leakSpan(), WTFMove(dataProvider), WTFMove(context)));
}

ImageBufferCGBitmapBackend::ImageBufferCGBitmapBackend(const Parameters& parameters, std::span<uint8_t> data, RetainPtr<CGDataProviderRef>&& dataProvider, std::unique_ptr<GraphicsContextCG>&& context)
    : ImageBufferCGBackend(parameters, WTFMove(context))
    , m_data(data)
    , m_dataProvider(WTFMove(dataProvider))
{
    ASSERT(m_data.data());
    ASSERT(m_dataProvider);
    ASSERT(m_context);
    applyBaseTransform(*m_context);
}

ImageBufferCGBitmapBackend::~ImageBufferCGBitmapBackend() = default;

GraphicsContext& ImageBufferCGBitmapBackend::context()
{
    return *m_context;
}

unsigned ImageBufferCGBitmapBackend::bytesPerRow() const
{
    return calculateBytesPerRow(m_parameters.backendSize);
}

bool ImageBufferCGBitmapBackend::canMapBackingStore() const
{
    return true;
}

RefPtr<NativeImage> ImageBufferCGBitmapBackend::copyNativeImage()
{
    return NativeImage::create(adoptCF(CGBitmapContextCreateImage(context().platformContext())));
}

RefPtr<NativeImage> ImageBufferCGBitmapBackend::createNativeImageReference()
{
    auto backendSize = size();
    return NativeImage::create(adoptCF(CGImageCreate(
        backendSize.width(), backendSize.height(), 8, 32, bytesPerRow(),
        colorSpace().platformColorSpace(), static_cast<uint32_t>(kCGImageAlphaPremultipliedFirst) | static_cast<uint32_t>(kCGBitmapByteOrder32Host), m_dataProvider.get(),
        0, true, kCGRenderingIntentDefault)));
}

void ImageBufferCGBitmapBackend::getPixelBuffer(const IntRect& srcRect, PixelBuffer& destination)
{
    ImageBufferBackend::getPixelBuffer(srcRect, m_data, destination);
}

void ImageBufferCGBitmapBackend::putPixelBuffer(const PixelBuffer& pixelBuffer, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat)
{
    ImageBufferBackend::putPixelBuffer(pixelBuffer, srcRect, destPoint, destFormat, m_data);
}

} // namespace WebCore

#endif // USE(CG)
