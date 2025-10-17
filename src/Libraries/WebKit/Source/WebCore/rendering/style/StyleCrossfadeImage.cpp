/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#include "StyleCrossfadeImage.h"

#include "AnimationUtilities.h"
#include "CSSCrossfadeValue.h"
#include "CSSValuePool.h"
#include "CachedImage.h"
#include "CachedResourceLoader.h"
#include "CrossfadeGeneratedImage.h"
#include "RenderElement.h"
#include <wtf/PointerComparison.h>

namespace WebCore {

StyleCrossfadeImage::StyleCrossfadeImage(RefPtr<StyleImage>&& from, RefPtr<StyleImage>&& to, double percentage, bool isPrefixed)
    : StyleGeneratedImage { Type::CrossfadeImage, StyleCrossfadeImage::isFixedSize }
    , m_from { WTFMove(from) }
    , m_to { WTFMove(to) }
    , m_percentage { percentage }
    , m_isPrefixed { isPrefixed }
    , m_inputImagesAreReady { false }
{
}

StyleCrossfadeImage::~StyleCrossfadeImage()
{
    if (m_cachedFromImage)
        m_cachedFromImage->removeClient(*this);
    if (m_cachedToImage)
        m_cachedToImage->removeClient(*this);
}

bool StyleCrossfadeImage::operator==(const StyleImage& other) const
{
    auto* otherCrossfadeImage = dynamicDowncast<StyleCrossfadeImage>(other);
    return otherCrossfadeImage && equals(*otherCrossfadeImage);
}

bool StyleCrossfadeImage::equals(const StyleCrossfadeImage& other) const
{
    return equalInputImages(other) && m_percentage == other.m_percentage;
}

bool StyleCrossfadeImage::equalInputImages(const StyleCrossfadeImage& other) const
{
    return arePointingToEqualData(m_from, other.m_from) && arePointingToEqualData(m_to, other.m_to);
}

RefPtr<StyleCrossfadeImage> StyleCrossfadeImage::blend(const StyleCrossfadeImage& from, const BlendingContext& context) const
{
    ASSERT(equalInputImages(from));

    if (!m_cachedToImage || !m_cachedFromImage)
        return nullptr;

    auto newPercentage = WebCore::blend(from.m_percentage, m_percentage, context);
    return StyleCrossfadeImage::create(m_from, m_to, newPercentage, from.m_isPrefixed && m_isPrefixed);
}

Ref<CSSValue> StyleCrossfadeImage::computedStyleValue(const RenderStyle& style) const
{
    auto fromComputedValue = m_from ? m_from->computedStyleValue(style) : static_reference_cast<CSSValue>(CSSPrimitiveValue::create(CSSValueNone));
    auto toComputedValue = m_to ? m_to->computedStyleValue(style) : static_reference_cast<CSSValue>(CSSPrimitiveValue::create(CSSValueNone));
    return CSSCrossfadeValue::create(WTFMove(fromComputedValue), WTFMove(toComputedValue), CSSPrimitiveValue::create(m_percentage), m_isPrefixed);
}

bool StyleCrossfadeImage::isPending() const
{
    if (m_from && m_from->isPending())
        return true;
    if (m_to && m_to->isPending())
        return true;
    return false;
}

void StyleCrossfadeImage::load(CachedResourceLoader& loader, const ResourceLoaderOptions& options)
{
    auto oldCachedFromImage = m_cachedFromImage;
    auto oldCachedToImage = m_cachedToImage;

    if (m_from) {
        if (m_from->isPending())
            m_from->load(loader, options);
        m_cachedFromImage = m_from->cachedImage();
    } else
        m_cachedFromImage = nullptr;

    if (m_to) {
        if (m_to->isPending())
            m_to->load(loader, options);
        m_cachedToImage = m_to->cachedImage();
    } else
        m_cachedToImage = nullptr;

    if (m_cachedFromImage != oldCachedFromImage) {
        if (oldCachedFromImage)
            oldCachedFromImage->removeClient(*this);
        if (m_cachedFromImage)
            m_cachedFromImage->addClient(*this);
    }

    if (m_cachedToImage != oldCachedToImage) {
        if (oldCachedToImage)
            oldCachedToImage->removeClient(*this);
        if (m_cachedToImage)
            m_cachedToImage->addClient(*this);
    }

    m_inputImagesAreReady = true;
}

RefPtr<Image> StyleCrossfadeImage::image(const RenderElement* renderer, const FloatSize& size, bool isForFirstLine) const
{
    if (!renderer)
        return &Image::nullImage();

    if (size.isEmpty())
        return nullptr;

    if (!m_from || !m_to)
        return &Image::nullImage();

    auto fromImage = m_from->image(renderer, size, isForFirstLine);
    auto toImage = m_to->image(renderer, size, isForFirstLine);

    if (!fromImage || !toImage)
        return &Image::nullImage();

    return CrossfadeGeneratedImage::create(*fromImage, *toImage, m_percentage, fixedSize(*renderer), size);
}

bool StyleCrossfadeImage::knownToBeOpaque(const RenderElement& renderer) const
{
    if (m_from && !m_from->knownToBeOpaque(renderer))
        return false;
    if (m_to && !m_to->knownToBeOpaque(renderer))
        return false;
    return true;
}

FloatSize StyleCrossfadeImage::fixedSize(const RenderElement& renderer) const
{
    if (!m_from || !m_to)
        return { };

    auto fromImageSize = m_from->imageSize(&renderer, 1);
    auto toImageSize = m_to->imageSize(&renderer, 1);

    // Rounding issues can cause transitions between images of equal size to return
    // a different fixed size; avoid performing the interpolation if the images are the same size.
    if (fromImageSize == toImageSize)
        return fromImageSize;

    float percentage = m_percentage;
    float inversePercentage = 1 - percentage;

    return fromImageSize * inversePercentage + toImageSize * percentage;
}

void StyleCrossfadeImage::imageChanged(CachedImage*, const IntRect*)
{
    if (!m_inputImagesAreReady)
        return;
    for (auto entry : clients()) {
        auto& client = entry.key;
        client.imageChanged(static_cast<WrappedImagePtr>(this));
    }
}

} // namespace WebCore
