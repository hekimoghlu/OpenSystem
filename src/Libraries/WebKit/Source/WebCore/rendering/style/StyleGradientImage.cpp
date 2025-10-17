/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#include "StyleGradientImage.h"

#include "CSSGradientValue.h"
#include "GeneratedImage.h"
#include "GradientImage.h"
#include "NodeRenderStyle.h"
#include "RenderElement.h"
#include "RenderStyleInlines.h"
#include "StylePrimitiveNumericTypes+Conversions.h"

namespace WebCore {

StyleGradientImage::StyleGradientImage(Style::Gradient&& gradient)
    : StyleGeneratedImage { Type::GradientImage, StyleGradientImage::isFixedSize }
    , m_gradient { WTFMove(gradient) }
    , m_knownCacheableBarringFilter { Style::stopsAreCacheable(m_gradient) }
{
}

StyleGradientImage::~StyleGradientImage() = default;

bool StyleGradientImage::operator==(const StyleImage& other) const
{
    auto* otherGradientImage = dynamicDowncast<StyleGradientImage>(other);
    return otherGradientImage && equals(*otherGradientImage);
}

bool StyleGradientImage::equals(const StyleGradientImage& other) const
{
    return m_gradient == other.m_gradient;
}

Ref<CSSValue> StyleGradientImage::computedStyleValue(const RenderStyle& style) const
{
    return CSSGradientValue::create(Style::toCSS(m_gradient, style));
}

bool StyleGradientImage::isPending() const
{
    return false;
}

void StyleGradientImage::load(CachedResourceLoader&, const ResourceLoaderOptions&)
{
}

RefPtr<Image> StyleGradientImage::image(const RenderElement* renderer, const FloatSize& size, bool isForFirstLine) const
{
    if (!renderer)
        return &Image::nullImage();

    if (size.isEmpty())
        return nullptr;

    auto& style = isForFirstLine ? renderer->firstLineStyle() : renderer->style();

    bool cacheable = m_knownCacheableBarringFilter && !style.hasAppleColorFilter();
    if (cacheable) {
        if (auto* result = const_cast<StyleGradientImage&>(*this).cachedImageForSize(size))
            return result;
    }

    auto gradient = Style::createPlatformGradient(m_gradient, size, style);

    auto newImage = GradientImage::create(WTFMove(gradient), size);
    if (cacheable)
        const_cast<StyleGradientImage&>(*this).saveCachedImageForSize(size, newImage);
    return newImage;
}

bool StyleGradientImage::knownToBeOpaque(const RenderElement& renderer) const
{
    return Style::isOpaque(m_gradient, renderer.style());
}

FloatSize StyleGradientImage::fixedSize(const RenderElement&) const
{
    return { };
}

} // namespace WebCore
