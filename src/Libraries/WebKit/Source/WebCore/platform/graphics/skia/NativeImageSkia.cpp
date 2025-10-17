/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#include "NativeImage.h"

#if USE(SKIA)
#include "GLContext.h"
#include "GLFence.h"
#include "GraphicsContextSkia.h"
#include "PlatformDisplay.h"
#include <skia/core/SkData.h>
#include <skia/core/SkImage.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN // GLib/Win ports
#include <skia/gpu/ganesh/GrBackendSurface.h>
#include <skia/gpu/ganesh/SkImageGanesh.h>
#include <skia/private/chromium/SkImageChromium.h>
#include <skia/core/SkPixmap.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

void PlatformImageNativeImageBackend::finishAcceleratedRenderingAndCreateFence()
{
    Locker locker { m_fenceLock };
    if (m_fence)
        return;

    auto* glContext = PlatformDisplay::sharedDisplay().skiaGLContext();
    if (!glContext || !glContext->makeContextCurrent())
        return;

    auto* grContext = PlatformDisplay::sharedDisplay().skiaGrContext();
    RELEASE_ASSERT(grContext);

    grContext->flush(m_platformImage);

    if (GLFence::isSupported()) {
        grContext->submit(GrSyncCpu::kNo);
        m_fence = GLFence::create();
    }

    if (!m_fence)
        grContext->submit(GrSyncCpu::kYes);
}

void PlatformImageNativeImageBackend::waitForAcceleratedRenderingFenceCompletion()
{
    Locker locker { m_fenceLock };
    if (!m_fence)
        return;

    m_fence->serverWait();
    m_fence = nullptr;
}

const GrDirectContext* PlatformImageNativeImageBackend::skiaGrContext() const
{
    return SkImages::GetContext(platformImage());
}

RefPtr<NativeImage> PlatformImageNativeImageBackend::copyAcceleratedNativeImageBorrowingBackendTexture() const
{
    auto image = platformImage();
    if (!image)
        return nullptr;

    auto* glContext = PlatformDisplay::sharedDisplay().skiaGLContext();
    if (!glContext || !glContext->makeContextCurrent())
        return nullptr;

    auto* grContext = PlatformDisplay::sharedDisplay().skiaGrContext();
    RELEASE_ASSERT(grContext);

    GrBackendTexture backendTexture;
    if (!SkImages::GetBackendTextureFromImage(image, &backendTexture, false))
        return nullptr;

    return NativeImage::create(SkImages::BorrowTextureFrom(grContext, backendTexture, kTopLeft_GrSurfaceOrigin, image->colorType(), image->alphaType(), image->refColorSpace()));
}

IntSize PlatformImageNativeImageBackend::size() const
{
    return m_platformImage ? IntSize(m_platformImage->width(), m_platformImage->height()) : IntSize();
}

bool PlatformImageNativeImageBackend::hasAlpha() const
{
    switch (m_platformImage->imageInfo().alphaType()) {
    case kUnknown_SkAlphaType:
    case kOpaque_SkAlphaType:
        return false;
    case kPremul_SkAlphaType:
    case kUnpremul_SkAlphaType:
        return true;
    }
    return false;
}

DestinationColorSpace PlatformImageNativeImageBackend::colorSpace() const
{
    if (auto colorSpace = platformImage()->refColorSpace())
        return DestinationColorSpace(colorSpace);
    // No color space means the default - SRGB.
    return DestinationColorSpace::SRGB();
}

Headroom PlatformImageNativeImageBackend::headroom() const
{
    return Headroom::None;
}

std::optional<Color> NativeImage::singlePixelSolidColor() const
{
    if (size() != IntSize(1, 1))
        return std::nullopt;

    auto platformImage = this->platformImage();
    if (platformImage->isTextureBacked()) {
        if (!PlatformDisplay::sharedDisplay().skiaGLContext()->makeContextCurrent())
            return std::nullopt;

        GrDirectContext* grContext = PlatformDisplay::sharedDisplay().skiaGrContext();
        const auto& imageInfo = platformImage->imageInfo();
        uint32_t pixel;
        SkPixmap pixmap(imageInfo, &pixel, imageInfo.minRowBytes());
        if (!platformImage->readPixels(grContext, pixmap, 0, 0))
            return std::nullopt;

        return pixmap.getColor(0, 0);
    }

    SkPixmap pixmap;
    if (!platformImage->peekPixels(&pixmap))
        return std::nullopt;

    return pixmap.getColor(0, 0);
}

void NativeImage::draw(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions options)
{
    context.drawNativeImageInternal(*this, destinationRect, sourceRect, options);
}

void NativeImage::clearSubimages()
{
}

#if USE(COORDINATED_GRAPHICS)
uint64_t NativeImage::uniqueID() const
{
    if (auto& image = platformImage())
        return image->uniqueID();
    return 0;
}
#endif

} // namespace WebCore

#endif // USE(SKIA)
