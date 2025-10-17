/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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
#include "StyleNamedImage.h"

#include "CSSNamedImageValue.h"
#include "NamedImageGeneratedImage.h"

namespace WebCore {

StyleNamedImage::StyleNamedImage(String&& name)
    : StyleGeneratedImage { Type::NamedImage, StyleNamedImage::isFixedSize }
    , m_name { WTFMove(name) }
{
}

StyleNamedImage::~StyleNamedImage() = default;

bool StyleNamedImage::operator==(const StyleImage& other) const
{
    auto* otherNamedImage = dynamicDowncast<StyleNamedImage>(other);
    return otherNamedImage && equals(*otherNamedImage);
}

bool StyleNamedImage::equals(const StyleNamedImage& other) const
{
    return m_name == other.m_name;
}

Ref<CSSValue> StyleNamedImage::computedStyleValue(const RenderStyle&) const
{
    return CSSNamedImageValue::create(m_name);
}

bool StyleNamedImage::isPending() const
{
    return false;
}

void StyleNamedImage::load(CachedResourceLoader&, const ResourceLoaderOptions&)
{
}

RefPtr<Image> StyleNamedImage::image(const RenderElement* renderer, const FloatSize& size, bool) const
{
    if (!renderer)
        return &Image::nullImage();

    if (size.isEmpty())
        return nullptr;

    return NamedImageGeneratedImage::create(m_name, size);
}

bool StyleNamedImage::knownToBeOpaque(const RenderElement&) const
{
    return false;
}

FloatSize StyleNamedImage::fixedSize(const RenderElement&) const
{
    return { };
}

} // namespace WebCore
