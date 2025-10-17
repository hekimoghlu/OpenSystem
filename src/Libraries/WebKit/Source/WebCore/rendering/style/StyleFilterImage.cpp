/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#include "StyleFilterImage.h"

#include "BitmapImage.h"
#include "CSSFilter.h"
#include "CSSFilterImageValue.h"
#include "CSSValuePool.h"
#include "CachedImage.h"
#include "CachedResourceLoader.h"
#include "ComputedStyleExtractor.h"
#include "HostWindow.h"
#include "ImageBuffer.h"
#include "NullGraphicsContext.h"
#include "RenderElement.h"
#include "StyleFilterProperty.h"
#include <wtf/PointerComparison.h>

namespace WebCore {

// MARK: - StyleFilterImage

StyleFilterImage::StyleFilterImage(RefPtr<StyleImage>&& image, FilterOperations&& filterOperations)
    : StyleGeneratedImage { Type::FilterImage, StyleFilterImage::isFixedSize }
    , m_image { WTFMove(image) }
    , m_filterOperations { WTFMove(filterOperations) }
    , m_inputImageIsReady { false }
{
}

StyleFilterImage::~StyleFilterImage()
{
    if (m_cachedImage)
        m_cachedImage->removeClient(*this);
}

bool StyleFilterImage::operator==(const StyleImage& other) const
{
    auto* otherFilterImage = dynamicDowncast<StyleFilterImage>(other);
    return otherFilterImage && equals(*otherFilterImage);
}

bool StyleFilterImage::equals(const StyleFilterImage& other) const
{
    return equalInputImages(other) && m_filterOperations == other.m_filterOperations;
}

bool StyleFilterImage::equalInputImages(const StyleFilterImage& other) const
{
    return arePointingToEqualData(m_image, other.m_image);
}

Ref<CSSValue> StyleFilterImage::computedStyleValue(const RenderStyle& style) const
{
    return CSSFilterImageValue::create(
        m_image ? m_image->computedStyleValue(style) : static_reference_cast<CSSValue>(CSSPrimitiveValue::create(CSSValueNone)),
        Style::toCSSFilterProperty(m_filterOperations, style)
    );
}

bool StyleFilterImage::isPending() const
{
    return m_image ? m_image->isPending() : false;
}

void StyleFilterImage::load(CachedResourceLoader& cachedResourceLoader, const ResourceLoaderOptions& options)
{
    CachedResourceHandle<CachedImage> oldCachedImage = m_cachedImage;

    if (m_image) {
        m_image->load(cachedResourceLoader, options);
        m_cachedImage = m_image->cachedImage();
    } else
        m_cachedImage = nullptr;

    if (m_cachedImage != oldCachedImage) {
        if (oldCachedImage)
            oldCachedImage->removeClient(*this);
        if (m_cachedImage)
            m_cachedImage->addClient(*this);
    }

    for (auto& filterOperation : m_filterOperations) {
        if (RefPtr referenceFilterOperation = dynamicDowncast<ReferenceFilterOperation>(filterOperation))
            referenceFilterOperation->loadExternalDocumentIfNeeded(cachedResourceLoader, options);
    }

    m_inputImageIsReady = true;
}

RefPtr<Image> StyleFilterImage::image(const RenderElement* renderer, const FloatSize& size, bool isForFirstLine) const
{
    if (!renderer)
        return &Image::nullImage();

    if (size.isEmpty())
        return nullptr;

    if (!m_image)
        return &Image::nullImage();

    auto image = m_image->image(renderer, size, isForFirstLine);
    if (!image || image->isNull())
        return &Image::nullImage();

    auto preferredFilterRenderingModes = renderer->page().preferredFilterRenderingModes();
    auto sourceImageRect = FloatRect { { }, size };

    auto cssFilter = CSSFilter::create(const_cast<RenderElement&>(*renderer), m_filterOperations, preferredFilterRenderingModes, FloatSize { 1, 1 }, sourceImageRect, NullGraphicsContext());
    if (!cssFilter)
        return &Image::nullImage();

    cssFilter->setFilterRegion(sourceImageRect);

    auto sourceImage = ImageBuffer::create(size, cssFilter->renderingMode(), RenderingPurpose::DOM, 1, DestinationColorSpace::SRGB(), ImageBufferPixelFormat::BGRA8, renderer->hostWindow());
    if (!sourceImage)
        return &Image::nullImage();

    auto filteredImage = sourceImage->filteredNativeImage(*cssFilter, [&](GraphicsContext& context) {
        context.drawImage(*image, sourceImageRect);
    });
    if (!filteredImage)
        return &Image::nullImage();
    return BitmapImage::create(WTFMove(filteredImage));
}

bool StyleFilterImage::knownToBeOpaque(const RenderElement&) const
{
    return false;
}

FloatSize StyleFilterImage::fixedSize(const RenderElement& renderer) const
{
    if (!m_image)
        return { };

    return m_image->imageSize(&renderer, 1);
}

void StyleFilterImage::imageChanged(CachedImage*, const IntRect*)
{
    if (!m_inputImageIsReady)
        return;

    for (auto entry : clients()) {
        auto& client = entry.key;
        client.imageChanged(static_cast<WrappedImagePtr>(this));
    }
}

} // namespace WebCore
