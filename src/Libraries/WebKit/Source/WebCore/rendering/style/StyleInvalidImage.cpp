/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
#include "StyleInvalidImage.h"

#include "CSSImageValue.h"

namespace WebCore {

Ref<StyleInvalidImage> StyleInvalidImage::create()
{
    return adoptRef(*new StyleInvalidImage);
}

StyleInvalidImage::~StyleInvalidImage() = default;

StyleInvalidImage::StyleInvalidImage()
    : StyleGeneratedImage(StyleImage::Type::InvalidImage, true)
{
}

void StyleInvalidImage::load(CachedResourceLoader&, const ResourceLoaderOptions&)
{
}

RefPtr<Image> StyleInvalidImage::image(const RenderElement*, const FloatSize&, bool) const
{
    return &Image::nullImage();
}

Ref<CSSValue> StyleInvalidImage::computedStyleValue(const RenderStyle&) const
{
    return CSSImageValue::create();
}

} // namespace WebCore

