/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#include "StyleMultiImage.h"

#include "CSSCanvasValue.h"
#include "CSSCrossfadeValue.h"
#include "CSSFilterImageValue.h"
#include "CSSGradientValue.h"
#include "CSSImageSetValue.h"
#include "CSSImageValue.h"
#include "CSSNamedImageValue.h"
#include "CSSPaintImageValue.h"
#include "CSSVariableData.h"
#include "CachedImage.h"
#include "CachedResourceLoader.h"
#include "RenderElement.h"
#include "RenderView.h"
#include "StyleCachedImage.h"
#include "StyleCanvasImage.h"
#include "StyleCrossfadeImage.h"
#include "StyleFilterImage.h"
#include "StyleGradientImage.h"
#include "StyleNamedImage.h"
#include "StylePaintImage.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(StyleMultiImage);

StyleMultiImage::StyleMultiImage(Type type)
    : StyleImage { type }
{
}

StyleMultiImage::~StyleMultiImage() = default;

bool StyleMultiImage::equals(const StyleMultiImage& other) const
{
    return (!m_isPending && !other.m_isPending && m_selectedImage.get() == other.m_selectedImage.get());
}

void StyleMultiImage::load(CachedResourceLoader& loader, const ResourceLoaderOptions& options)
{
    ASSERT(m_isPending);
    ASSERT(loader.document());

    m_isPending = false;

    auto bestFitImage = selectBestFitImage(*loader.document());

    ASSERT(is<StyleCachedImage>(bestFitImage.image) || is<StyleGeneratedImage>(bestFitImage.image));

    if (is<StyleGeneratedImage>(bestFitImage.image)) {
        m_selectedImage = bestFitImage.image;
        m_selectedImage->load(loader, options);
        return;
    }
    
    if (RefPtr styleCachedImage = dynamicDowncast<StyleCachedImage>(bestFitImage.image)) {
        if (styleCachedImage->imageScaleFactor() == bestFitImage.scaleFactor)
            m_selectedImage = WTFMove(styleCachedImage);
        else
            m_selectedImage = StyleCachedImage::copyOverridingScaleFactor(*styleCachedImage, bestFitImage.scaleFactor);

        if (m_selectedImage->isPending())
            m_selectedImage->load(loader, options);
        return;
    }
}

CachedImage* StyleMultiImage::cachedImage() const
{
    if (!m_selectedImage)
        return nullptr;
    return m_selectedImage->cachedImage();
}

WrappedImagePtr StyleMultiImage::data() const
{
    if (!m_selectedImage)
        return nullptr;
    return m_selectedImage->data();
}

bool StyleMultiImage::canRender(const RenderElement* renderer, float multiplier) const
{
    return m_selectedImage && m_selectedImage->canRender(renderer, multiplier);
}

bool StyleMultiImage::isPending() const
{
    return m_isPending;
}

bool StyleMultiImage::isLoaded(const RenderElement* renderer) const
{
    return m_selectedImage && m_selectedImage->isLoaded(renderer);
}

bool StyleMultiImage::errorOccurred() const
{
    return m_selectedImage && m_selectedImage->errorOccurred();
}

FloatSize StyleMultiImage::imageSize(const RenderElement* renderer, float multiplier) const
{
    if (!m_selectedImage)
        return { };
    return m_selectedImage->imageSize(renderer, multiplier);
}

bool StyleMultiImage::imageHasRelativeWidth() const
{
    return m_selectedImage && m_selectedImage->imageHasRelativeWidth();
}

bool StyleMultiImage::imageHasRelativeHeight() const
{
    return m_selectedImage && m_selectedImage->imageHasRelativeHeight();
}

void StyleMultiImage::computeIntrinsicDimensions(const RenderElement* element, Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio)
{
    if (!m_selectedImage)
        return;
    m_selectedImage->computeIntrinsicDimensions(element, intrinsicWidth, intrinsicHeight, intrinsicRatio);
}

bool StyleMultiImage::usesImageContainerSize() const
{
    return m_selectedImage && m_selectedImage->usesImageContainerSize();
}

void StyleMultiImage::setContainerContextForRenderer(const RenderElement& renderer, const FloatSize& containerSize, float containerZoom)
{
    if (!m_selectedImage)
        return;
    m_selectedImage->setContainerContextForRenderer(renderer, containerSize, containerZoom);
}

void StyleMultiImage::addClient(RenderElement& renderer)
{
    if (!m_selectedImage)
        return;
    m_selectedImage->addClient(renderer);
}

void StyleMultiImage::removeClient(RenderElement& renderer)
{
    if (!m_selectedImage)
        return;
    m_selectedImage->removeClient(renderer);
}

bool StyleMultiImage::hasClient(RenderElement& renderer) const
{
    if (!m_selectedImage)
        return false;
    return m_selectedImage->hasClient(renderer);
}

RefPtr<Image> StyleMultiImage::image(const RenderElement* renderer, const FloatSize& size, bool isForFirstLine) const
{
    if (!m_selectedImage)
        return nullptr;
    return m_selectedImage->image(renderer, size, isForFirstLine);
}

float StyleMultiImage::imageScaleFactor() const
{
    if (!m_selectedImage)
        return 1;
    return m_selectedImage->imageScaleFactor();
}

bool StyleMultiImage::knownToBeOpaque(const RenderElement& renderer) const
{
    return m_selectedImage && m_selectedImage->knownToBeOpaque(renderer);
}

} // namespace WebCore
