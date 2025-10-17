/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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
#include "StyleCursorImage.h"

#include "CSSCursorImageValue.h"
#include "CSSImageValue.h"
#include "CSSValuePair.h"
#include "CachedImage.h"
#include "FloatSize.h"
#include "RenderElement.h"
#include "SVGCursorElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGLengthContext.h"
#include "SVGURIReference.h"
#include "StyleBuilderState.h"
#include "StyleCachedImage.h"
#include "StyleImageSet.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(StyleCursorImage);

Ref<StyleCursorImage> StyleCursorImage::create(Ref<StyleImage>&& image, const std::optional<IntPoint>& hotSpot, const URL& originalURL, LoadedFromOpaqueSource loadedFromOpaqueSource)
{ 
    return adoptRef(*new StyleCursorImage(WTFMove(image), hotSpot, originalURL, loadedFromOpaqueSource));
}

StyleCursorImage::StyleCursorImage(Ref<StyleImage>&& image, const std::optional<IntPoint>& hotSpot, const URL& originalURL, LoadedFromOpaqueSource loadedFromOpaqueSource)
    : StyleMultiImage { Type::CursorImage }
    , m_image { WTFMove(image) }
    , m_hotSpot { hotSpot }
    , m_originalURL { originalURL }
    , m_loadedFromOpaqueSource { loadedFromOpaqueSource }
{
}

StyleCursorImage::~StyleCursorImage()
{
    for (auto& element : m_cursorElements)
        element.removeClient(*this);
}

bool StyleCursorImage::operator==(const StyleImage& other) const
{
    auto* otherCursorImage = dynamicDowncast<StyleCursorImage>(other);
    return otherCursorImage && equals(*otherCursorImage);
}

bool StyleCursorImage::equals(const StyleCursorImage& other) const
{
    return equalInputImages(other) && StyleMultiImage::equals(other);
}

bool StyleCursorImage::equalInputImages(const StyleCursorImage& other) const
{
    return m_image.get() == other.m_image.get();
}

Ref<CSSValue> StyleCursorImage::computedStyleValue(const RenderStyle& style) const
{
    RefPtr<CSSValuePair> hotSpot;
    if (m_hotSpot)
        hotSpot = CSSValuePair::createNoncoalescing(CSSPrimitiveValue::create(m_hotSpot->x()), CSSPrimitiveValue::create(m_hotSpot->y()));

    return CSSCursorImageValue::create(m_image->computedStyleValue(style), WTFMove(hotSpot), m_originalURL, m_loadedFromOpaqueSource );
}

ImageWithScale StyleCursorImage::selectBestFitImage(const Document& document)
{
    if (RefPtr imageSet = dynamicDowncast<StyleImageSet>(m_image.get()))
        return imageSet->selectBestFitImage(document);

    if (RefPtr cachedImage = dynamicDowncast<StyleCachedImage>(m_image.get())) {
        if (RefPtr cursorElement = updateCursorElement(document)) {
            auto existingImageURL = cachedImage->imageURL();
            auto updatedImageURL = document.completeURL(cursorElement->href());

            if (existingImageURL != updatedImageURL)
                m_image = StyleCachedImage::create(CSSImageValue::create(WTFMove(updatedImageURL), m_loadedFromOpaqueSource));
        }
    }

    return { m_image.ptr(), 1, String() };
}

RefPtr<SVGCursorElement> StyleCursorImage::updateCursorElement(const Document& document)
{
    RefPtr cursorElement = dynamicDowncast<SVGCursorElement>(SVGURIReference::targetElementFromIRIString(m_originalURL.string(), document).element);
    if (!cursorElement)
        return nullptr;

    // FIXME: Not right to keep old cursor elements as clients. The new one should replace the old, not join it in a set.
    if (m_cursorElements.add(*cursorElement).isNewEntry) {
        cursorElementChanged(*cursorElement);
        cursorElement->addClient(*this);
    }
    return cursorElement;
}

void StyleCursorImage::cursorElementRemoved(SVGCursorElement& cursorElement)
{
    // FIXME: Not right to stay a client of a cursor element until the element is destroyed. We'd want to stop being a client once it's no longer a valid target, like when it's disconnected.
    m_cursorElements.remove(cursorElement);
}

void StyleCursorImage::cursorElementChanged(SVGCursorElement& cursorElement)
{
    // FIXME: Seems wrong that changing an old cursor element, one that that is no longer the target, changes the hot spot.
    // FIXME: This will override a hot spot that was specified in CSS, which is probably incorrect.
    // FIXME: Should we clamp from float to int instead of just casting here?
    SVGLengthContext lengthContext(nullptr);
    m_hotSpot = IntPoint {
        static_cast<int>(std::round(cursorElement.x().value(lengthContext))),
        static_cast<int>(std::round(cursorElement.y().value(lengthContext)))
    };

    // FIXME: Why doesn't this funtion check for a change to the href of the cursor element? Why would we dynamically track changes to x/y but not href?
}

void StyleCursorImage::setContainerContextForRenderer(const RenderElement& renderer, const FloatSize& containerSize, float containerZoom)
{
    if (!hasCachedImage())
        return;
    cachedImage()->setContainerContextForClient(renderer, LayoutSize(containerSize), containerZoom, m_originalURL);
}

bool StyleCursorImage::usesDataProtocol() const
{
    return m_originalURL.protocolIsData();
}

} // namespace WebCore
