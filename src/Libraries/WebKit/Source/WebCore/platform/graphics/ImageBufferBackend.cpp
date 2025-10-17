/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
#include "ImageBufferBackend.h"

#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "PixelBuffer.h"
#include "PixelBufferConversion.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ThreadSafeImageBufferFlusher);

IntSize ImageBufferBackend::calculateSafeBackendSize(const Parameters& parameters)
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

size_t ImageBufferBackend::calculateMemoryCost(const IntSize& backendSize, unsigned bytesPerRow)
{
    ASSERT(!backendSize.isEmpty());
    return CheckedUint32(backendSize.height()) * bytesPerRow;
}

ImageBufferBackend::ImageBufferBackend(const Parameters& parameters)
    : m_parameters(parameters)
{
}

ImageBufferBackend::~ImageBufferBackend() = default;

RefPtr<NativeImage> ImageBufferBackend::sinkIntoNativeImage()
{
    return createNativeImageReference();
}

void ImageBufferBackend::convertToLuminanceMask()
{
    IntRect sourceRect { { }, size() };
    PixelBufferFormat format { AlphaPremultiplication::Unpremultiplied, PixelFormat::RGBA8, colorSpace() };
    auto pixelBuffer = ImageBufferAllocator().createPixelBuffer(format, sourceRect.size());
    if (!pixelBuffer)
        return;
    getPixelBuffer(sourceRect, *pixelBuffer);

    unsigned pixelArrayLength = pixelBuffer->bytes().size();
    for (unsigned pixelOffset = 0; pixelOffset < pixelArrayLength; pixelOffset += 4) {
        uint8_t a = pixelBuffer->item(pixelOffset + 3);
        if (!a)
            continue;
        uint8_t r = pixelBuffer->item(pixelOffset);
        uint8_t g = pixelBuffer->item(pixelOffset + 1);
        uint8_t b = pixelBuffer->item(pixelOffset + 2);

        double luma = (r * 0.2125 + g * 0.7154 + b * 0.0721) * ((double)a / 255.0);
        pixelBuffer->set(pixelOffset + 3, luma);
    }

    putPixelBuffer(*pixelBuffer, sourceRect, IntPoint::zero(), AlphaPremultiplication::Premultiplied);
}

void ImageBufferBackend::getPixelBuffer(const IntRect& sourceRect, std::span<const uint8_t> sourceData, PixelBuffer& destinationPixelBuffer)
{
    IntRect backendRect { { }, size() };
    auto sourceRectClipped = intersection(backendRect, sourceRect);
    IntRect destinationRect { IntPoint::zero(), sourceRectClipped.size() };

    if (sourceRect.x() < 0)
        destinationRect.setX(-sourceRect.x());

    if (sourceRect.y() < 0)
        destinationRect.setY(-sourceRect.y());

    if (destinationRect.size() != sourceRect.size())
        destinationPixelBuffer.zeroFill();

    unsigned sourceBytesPerRow = bytesPerRow();
    ConstPixelBufferConversionView source {
        { AlphaPremultiplication::Premultiplied, convertToPixelFormat(pixelFormat()), colorSpace() },
        sourceBytesPerRow,
        sourceData.subspan(sourceRectClipped.y() * sourceBytesPerRow + sourceRectClipped.x() * 4)
    };
    unsigned destinationBytesPerRow = static_cast<unsigned>(4u * sourceRect.width());
    size_t offset = destinationRect.y() * destinationBytesPerRow + destinationRect.x() * 4;
    if (offset > destinationPixelBuffer.bytes().size())
        return;

    PixelBufferConversionView destination {
        destinationPixelBuffer.format(),
        destinationBytesPerRow,
        destinationPixelBuffer.bytes().subspan(offset)
    };

    convertImagePixels(source, destination, destinationRect.size());
}

void ImageBufferBackend::putPixelBuffer(const PixelBuffer& sourcePixelBuffer, const IntRect& sourceRect, const IntPoint& destinationPoint, AlphaPremultiplication destinationAlphaFormat, std::span<uint8_t> destinationData)
{
    IntRect backendRect { { }, size() };
    auto sourceRectClipped = intersection({ IntPoint::zero(), sourcePixelBuffer.size() }, sourceRect);
    auto destinationRect = sourceRectClipped;
    destinationRect.moveBy(destinationPoint);

    if (sourceRect.x() < 0)
        destinationRect.setX(destinationRect.x() - sourceRect.x());

    if (sourceRect.y() < 0)
        destinationRect.setY(destinationRect.y() - sourceRect.y());

    destinationRect.intersect(backendRect);
    sourceRectClipped.setSize(destinationRect.size());

    unsigned sourceBytesPerRow = static_cast<unsigned>(4u * sourcePixelBuffer.size().width());
    ConstPixelBufferConversionView source {
        sourcePixelBuffer.format(),
        sourceBytesPerRow,
        sourcePixelBuffer.bytes().subspan(sourceRectClipped.y() * sourceBytesPerRow + sourceRectClipped.x() * 4)
    };
    unsigned destinationBytesPerRow = bytesPerRow();
    PixelBufferConversionView destination {
        { destinationAlphaFormat, convertToPixelFormat(pixelFormat()), colorSpace() },
        destinationBytesPerRow,
        destinationData.subspan(destinationRect.y() * destinationBytesPerRow + destinationRect.x() * 4)
    };

    convertImagePixels(source, destination, destinationRect.size());
}

RefPtr<SharedBuffer> ImageBufferBackend::sinkIntoPDFDocument()
{
    return nullptr;
}

AffineTransform ImageBufferBackend::calculateBaseTransform(const Parameters& parameters)
{
    AffineTransform baseTransform;
#if USE(CG)
    // CoreGraphics origin is at bottom left corner. GraphicsContext origin is at top left corner. Flip the drawing with GraphicsContext base
    // transform.
    baseTransform.scale(1, -1);
    baseTransform.translate(0, -parameters.backendSize.height());
#endif
    baseTransform.scale(parameters.resolutionScale);
    return baseTransform;
}

#if USE(SKIA)
RefPtr<ImageBuffer> ImageBufferBackend::copyAcceleratedImageBufferBorrowingBackendRenderTarget(const ImageBuffer&) const
{
    return nullptr;
}
#endif

TextStream& operator<<(TextStream& ts, VolatilityState state)
{
    switch (state) {
    case VolatilityState::NonVolatile: ts << "non-volatile"; break;
    case VolatilityState::Volatile: ts << "volatile"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, const ImageBufferBackend& imageBufferBackend)
{
    ts << imageBufferBackend.debugDescription();
    return ts;
}

} // namespace WebCore
