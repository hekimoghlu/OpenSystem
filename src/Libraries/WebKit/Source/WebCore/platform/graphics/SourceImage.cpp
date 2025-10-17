/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#include "SourceImage.h"

#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "NativeImage.h"

namespace WebCore {

SourceImage::SourceImage(ImageVariant&& imageVariant)
    : m_imageVariant(WTFMove(imageVariant))
{
}

SourceImage::SourceImage(const SourceImage&) = default;
SourceImage::SourceImage(SourceImage&&) = default;
SourceImage& SourceImage::operator=(const SourceImage&) = default;
SourceImage& SourceImage::operator=(SourceImage&&) = default;
SourceImage::~SourceImage() = default;

bool SourceImage::operator==(const SourceImage& other) const
{
    return imageIdentifier() == other.imageIdentifier();
}

static inline NativeImage* nativeImageOf(const SourceImage::ImageVariant& imageVariant)
{
    if (auto* nativeImage = std::get_if<Ref<NativeImage>>(&imageVariant))
        return nativeImage->ptr();
    return nullptr;
}

NativeImage* SourceImage::nativeImageIfExists() const
{
    return nativeImageOf(m_imageVariant);
}

NativeImage* SourceImage::nativeImage() const
{
    if (!std::holds_alternative<Ref<ImageBuffer>>(m_imageVariant))
        return nativeImageIfExists();

    if (!m_transformedImageVariant) {
        auto imageBuffer = std::get<Ref<ImageBuffer>>(m_imageVariant);

        auto nativeImage = imageBuffer->createNativeImageReference();
        if (!nativeImage)
            return nullptr;

        m_transformedImageVariant = { nativeImage.releaseNonNull() };
    }

    ASSERT(m_transformedImageVariant);
    return nativeImageOf(*m_transformedImageVariant);
}

static inline ImageBuffer* imageBufferOf(const SourceImage::ImageVariant& imageVariant)
{
    if (auto* imageBuffer = std::get_if<Ref<ImageBuffer>>(&imageVariant))
        return imageBuffer->ptr();
    return nullptr;
}

ImageBuffer* SourceImage::imageBufferIfExists() const
{
    return imageBufferOf(m_imageVariant);
}

ImageBuffer* SourceImage::imageBuffer() const
{
    if (!std::holds_alternative<Ref<NativeImage>>(m_imageVariant))
        return imageBufferIfExists();

    if (!m_transformedImageVariant) {
        auto nativeImage = std::get<Ref<NativeImage>>(m_imageVariant);

        auto rect = FloatRect { { }, nativeImage->size() };
        auto imageBuffer = ImageBuffer::create(nativeImage->size(), RenderingMode::Unaccelerated, RenderingPurpose::Unspecified, 1, DestinationColorSpace::SRGB(), ImageBufferPixelFormat::BGRA8);
        if (!imageBuffer)
            return nullptr;

        imageBuffer->context().drawNativeImage(nativeImage, rect, rect);
        m_transformedImageVariant = { imageBuffer.releaseNonNull() };
    }

    ASSERT(m_transformedImageVariant);
    return imageBufferOf(*m_transformedImageVariant);
}

RenderingResourceIdentifier SourceImage::imageIdentifier() const
{
    return WTF::switchOn(m_imageVariant,
        [&] (const Ref<NativeImage>& nativeImage) {
            return nativeImage->renderingResourceIdentifier();
        },
        [&] (const Ref<ImageBuffer>& imageBuffer) {
            return imageBuffer->renderingResourceIdentifier();
        },
        [&] (RenderingResourceIdentifier renderingResourceIdentifier) {
            return renderingResourceIdentifier;
        }
    );
}

IntSize SourceImage::size() const
{
    return WTF::switchOn(m_imageVariant,
        [&] (const Ref<NativeImage>& nativeImage) {
            return nativeImage->size();
        },
        [&] (const Ref<ImageBuffer>& imageBuffer) {
            return imageBuffer->backendSize();
        },
        [&] (RenderingResourceIdentifier) -> IntSize {
            ASSERT_NOT_REACHED();
            return { };
        }
    );
}

} // namespace WebCore
