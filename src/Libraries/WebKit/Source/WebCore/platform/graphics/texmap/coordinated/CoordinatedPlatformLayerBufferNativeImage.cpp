/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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
#include "CoordinatedPlatformLayerBufferNativeImage.h"

#if USE(COORDINATED_GRAPHICS)
#include "BitmapTexture.h"
#include "CoordinatedPlatformLayerBufferRGB.h"
#include "NativeImage.h"
#include "TextureMapper.h"

#if USE(CAIRO)
#include <cairo.h>
#endif

#if USE(SKIA)
#include "GLContext.h"
#include "PlatformDisplay.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/gpu/ganesh/GrBackendSurface.h>
#include <skia/gpu/ganesh/SkImageGanesh.h>
#include <skia/gpu/ganesh/gl/GrGLBackendSurface.h>
#include <skia/core/SkPixmap.h> // NOLINT
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#endif

namespace WebCore {

std::unique_ptr<CoordinatedPlatformLayerBufferNativeImage> CoordinatedPlatformLayerBufferNativeImage::create(Ref<NativeImage>&& nativeImage, std::unique_ptr<GLFence>&& fence)
{
    OptionSet<TextureMapperFlags> flags;
    if (nativeImage->hasAlpha())
        flags.add(TextureMapperFlags::ShouldBlend);
    return makeUnique<CoordinatedPlatformLayerBufferNativeImage>(WTFMove(nativeImage), flags, WTFMove(fence));
}

CoordinatedPlatformLayerBufferNativeImage::CoordinatedPlatformLayerBufferNativeImage(Ref<NativeImage>&& nativeImage, OptionSet<TextureMapperFlags> flags, std::unique_ptr<GLFence>&& fence)
    : CoordinatedPlatformLayerBuffer(Type::NativeImage, nativeImage->size(), flags, WTFMove(fence))
    , m_image(WTFMove(nativeImage))
{
#if USE(SKIA)
    const auto& image = m_image->platformImage();
    if (!image->isTextureBacked())
        return;

    auto& display = PlatformDisplay::sharedDisplay();
    if (!display.skiaGLContext()->makeContextCurrent())
        return;

    auto* grContext = display.skiaGrContext();
    RELEASE_ASSERT(grContext);
    grContext->flushAndSubmit(GLFence::isSupported() ? GrSyncCpu::kNo : GrSyncCpu::kYes);

    unsigned textureID = 0;
    GrBackendTexture backendTexture;
    if (SkImages::GetBackendTextureFromImage(image, &backendTexture, false)) {
        GrGLTextureInfo textureInfo;
        if (GrBackendTextures::GetGLTextureInfo(backendTexture, &textureInfo))
            textureID = textureInfo.fID;
    }
    if (!textureID)
        return;

    m_buffer = CoordinatedPlatformLayerBufferRGB::create(textureID, m_image->size(), m_flags, GLFence::create());
#endif
}

CoordinatedPlatformLayerBufferNativeImage::~CoordinatedPlatformLayerBufferNativeImage() = default;

bool CoordinatedPlatformLayerBufferNativeImage::tryEnsureBuffer(TextureMapper& textureMapper)
{
    if (m_buffer)
        return true;

#if USE(SKIA)
    if (m_image->platformImage()->isTextureBacked())
        return false;
#endif

    OptionSet<BitmapTexture::Flags> textureFlags;
    if (m_image->hasAlpha())
        textureFlags.add(BitmapTexture::Flags::SupportsAlpha);
    auto texture = textureMapper.acquireTextureFromPool(m_size, textureFlags);

#if USE(CAIRO)
    auto* surface = m_image->platformImage().get();
    auto* imageData = cairo_image_surface_get_data(surface);
    texture->updateContents(imageData, IntRect(IntPoint(), m_size), IntPoint(), cairo_image_surface_get_stride(surface), PixelFormat::BGRA8);
#elif USE(SKIA)
    const auto& image = m_image->platformImage();
    SkPixmap pixmap;
    if (image->peekPixels(&pixmap))
        texture->updateContents(pixmap.addr(), IntRect(IntPoint(), m_size), IntPoint(), image->imageInfo().minRowBytes(), PixelFormat::BGRA8);
#endif

    m_buffer = CoordinatedPlatformLayerBufferRGB::create(WTFMove(texture), m_flags, nullptr);
    return true;
}

void CoordinatedPlatformLayerBufferNativeImage::paintToTextureMapper(TextureMapper& textureMapper, const FloatRect& targetRect, const TransformationMatrix& modelViewMatrix, float opacity)
{
    waitForContentsIfNeeded();

    if (!tryEnsureBuffer(textureMapper))
        return;

    m_buffer->paintToTextureMapper(textureMapper, targetRect, modelViewMatrix, opacity);
}

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
