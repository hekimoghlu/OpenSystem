/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#include "StyleImageSet.h"

#include "CSSImageSetOptionValue.h"
#include "CSSImageSetValue.h"
#include "CSSPrimitiveValue.h"
#include "Document.h"
#include "MIMETypeRegistry.h"
#include "Page.h"
#include "StyleInvalidImage.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(StyleImageSet);

Ref<StyleImageSet> StyleImageSet::create(Vector<ImageWithScale>&& images, Vector<size_t>&& sortedIndices)
{
    ASSERT(images.size() == sortedIndices.size());
    return adoptRef(*new StyleImageSet(WTFMove(images), WTFMove(sortedIndices)));
}

StyleImageSet::StyleImageSet(Vector<ImageWithScale>&& images, Vector<size_t>&& sortedIndices)
    : StyleMultiImage { Type::ImageSet }
    , m_images { WTFMove(images) }
    , m_sortedIndices { WTFMove(sortedIndices) }
{
}

StyleImageSet::~StyleImageSet() = default;

bool StyleImageSet::operator==(const StyleImage& other) const
{
    auto* otherImageSet = dynamicDowncast<StyleImageSet>(other);
    return otherImageSet && equals(*otherImageSet);
}

bool StyleImageSet::equals(const StyleImageSet& other) const
{
    return m_images == other.m_images && StyleMultiImage::equals(other);
}

Ref<CSSValue> StyleImageSet::computedStyleValue(const RenderStyle& style) const
{
    auto builder = WTF::map<CSSValueListBuilderInlineCapacity>(m_images, [&](auto& image) -> Ref<CSSValue> {
        return CSSImageSetOptionValue::create(image.image->computedStyleValue(style), CSSPrimitiveValue::create(image.scaleFactor, CSSUnitType::CSS_DPPX), image.mimeType);
    });
    return CSSImageSetValue::create(WTFMove(builder));
}

ImageWithScale StyleImageSet::selectBestFitImage(const Document& document)
{
    updateDeviceScaleFactor(document);

    if (!m_accessedBestFitImage) {
        m_accessedBestFitImage = true;
        m_bestFitImage = bestImageForScaleFactor();
    }

    return m_bestFitImage;
}

ImageWithScale StyleImageSet::bestImageForScaleFactor()
{
    ImageWithScale result;
    for (auto index : m_sortedIndices) {
        const auto& image = m_images[index];
        if (!image.mimeType.isNull() && !MIMETypeRegistry::isSupportedImageMIMEType(image.mimeType))
            continue;
        if (!result.image->isInvalidImage() && result.scaleFactor == image.scaleFactor)
            continue;
        if (image.scaleFactor >= m_deviceScaleFactor)
            return image;

        result = image;
    }

    ASSERT(result.scaleFactor >= 0);
    if (result.image->isInvalidImage() || !result.scaleFactor)
        result = ImageWithScale { StyleInvalidImage::create(), 1, String() };

    return result;
}

void StyleImageSet::updateDeviceScaleFactor(const Document& document)
{
    // FIXME: In the future, we want to take much more than deviceScaleFactor into acount here.
    // All forms of scale should be included: Page::pageScaleFactor(), Frame::pageZoomFactor(),
    // and any CSS transforms. https://bugs.webkit.org/show_bug.cgi?id=81698
    float deviceScaleFactor = document.page() ? document.page()->deviceScaleFactor() : 1;
    if (deviceScaleFactor == m_deviceScaleFactor)
        return;
    m_deviceScaleFactor = deviceScaleFactor;
    m_accessedBestFitImage = false;
}

} // namespace WebCore
