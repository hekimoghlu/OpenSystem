/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
#include "GraphicsContextGL.h"

#if ENABLE(WEBGL) && USE(SKIA)
#include "BitmapImage.h"
#include "GLContext.h"
#include "GraphicsContextGLImageExtractor.h"
#include "NotImplemented.h"
#include "PixelBuffer.h"
#include "PlatformDisplay.h"
#include "SharedBuffer.h"
#include "SkiaSpanExtras.h"
#include <skia/core/SkData.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkImage.h>
#include <skia/core/SkPixmap.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

GraphicsContextGLImageExtractor::~GraphicsContextGLImageExtractor() = default;

bool GraphicsContextGLImageExtractor::extractImage(bool premultiplyAlpha, bool ignoreGammaAndColorProfile, bool ignoreNativeImageAlphaPremultiplication)
{
    PlatformImagePtr platformImage;
    bool hasAlpha = !m_image->currentFrameKnownToBeOpaque();
    if ((ignoreGammaAndColorProfile || (hasAlpha && !premultiplyAlpha)) && m_image->data()) {
        auto image = BitmapImage::create(nullptr,  AlphaOption::NotPremultiplied, ignoreGammaAndColorProfile ? GammaAndColorProfileOption::Ignored : GammaAndColorProfileOption::Applied);
        image->setData(m_image->data(), true);
        if (!image->frameCount())
            return false;

        platformImage = image->currentNativeImage()->platformImage();
    } else
        platformImage = m_image->currentNativeImage()->platformImage();

    if (!platformImage)
        return false;

    m_imageWidth = platformImage->width();
    m_imageHeight = platformImage->height();
    if (!m_imageWidth || !m_imageHeight)
        return false;

    const auto& imageInfo = platformImage->imageInfo();
    m_alphaOp = AlphaOp::DoNothing;
    switch (imageInfo.alphaType()) {
    case kUnknown_SkAlphaType:
    case kOpaque_SkAlphaType:
        break;
    case kPremul_SkAlphaType:
        if (!premultiplyAlpha)
            m_alphaOp = AlphaOp::DoUnmultiply;
        else if (ignoreNativeImageAlphaPremultiplication)
            m_alphaOp = AlphaOp::DoPremultiply;
        break;
    case kUnpremul_SkAlphaType:
        if (premultiplyAlpha)
            m_alphaOp = AlphaOp::DoPremultiply;
        break;
    }

    unsigned srcUnpackAlignment = 1;
    size_t bytesPerRow = imageInfo.minRowBytes();
    size_t bytesPerPixel = imageInfo.bytesPerPixel();
    unsigned padding = bytesPerRow - bytesPerPixel * m_imageWidth;
    if (padding) {
        srcUnpackAlignment = padding + 1;
        while (bytesPerRow % srcUnpackAlignment)
            ++srcUnpackAlignment;
    }

    if (platformImage->isTextureBacked()) {
        auto data = SkData::MakeUninitialized(imageInfo.computeMinByteSize());
        if (!PlatformDisplay::sharedDisplay().skiaGLContext()->makeContextCurrent())
            return false;

        GrDirectContext* grContext = PlatformDisplay::sharedDisplay().skiaGrContext();
        if (!platformImage->readPixels(grContext, imageInfo, static_cast<uint8_t*>(data->writable_data()), bytesPerRow, 0, 0))
            return false;

        m_pixelData = WTFMove(data);
        m_imagePixelData = span(m_pixelData.get());

        // SkSurfaces backed by textures have RGBA format.
        m_imageSourceFormat = DataFormat::RGBA8;
    } else {
        SkPixmap pixmap;
        if (!platformImage->peekPixels(&pixmap))
            return false;

        m_skImage = WTFMove(platformImage);
        m_imagePixelData = span(pixmap);

        // Raster SkSurfaces have BGRA format.
        m_imageSourceFormat = DataFormat::BGRA8;
    }

    m_imageSourceUnpackAlignment = srcUnpackAlignment;
    return true;
}

RefPtr<NativeImage> GraphicsContextGL::createNativeImageFromPixelBuffer(const GraphicsContextGLAttributes& sourceContextAttributes, Ref<PixelBuffer>&& pixelBuffer)
{
    ASSERT(!pixelBuffer->size().isEmpty());
    auto imageSize = pixelBuffer->size();
    SkAlphaType alphaType = kUnpremul_SkAlphaType;
    if (!sourceContextAttributes.alpha)
        alphaType = kOpaque_SkAlphaType;
    else if (sourceContextAttributes.premultipliedAlpha)
        alphaType = kPremul_SkAlphaType;
    auto imageInfo = SkImageInfo::Make(imageSize.width(), imageSize.height(), kRGBA_8888_SkColorType, alphaType, SkColorSpace::MakeSRGB());

    Ref protectedPixelBuffer = pixelBuffer;
    SkPixmap pixmap(imageInfo, pixelBuffer->bytes().data(), imageInfo.minRowBytes());
    auto image = SkImages::RasterFromPixmap(pixmap, [](const void*, void* context) {
        static_cast<PixelBuffer*>(context)->deref();
    }, &protectedPixelBuffer.leakRef());
    return NativeImage::create(WTFMove(image));
}

} // namespace WebCore

#endif // ENABLE(WEBGL) && USE(SKIA)
