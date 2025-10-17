/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#include "StyleCanvasImage.h"

#include "CSSCanvasValue.h"
#include "HTMLCanvasElement.h"
#include "InspectorInstrumentation.h"
#include "RenderElement.h"

namespace WebCore {

StyleCanvasImage::StyleCanvasImage(String&& name)
    : StyleGeneratedImage { Type::CanvasImage, StyleCanvasImage::isFixedSize }
    , m_name { WTFMove(name) }
{
}

StyleCanvasImage::~StyleCanvasImage()
{
    if (m_element)
        m_element->removeObserver(*this);
}

bool StyleCanvasImage::operator==(const StyleImage& other) const
{
    auto* otherCanvasImage = dynamicDowncast<StyleCanvasImage>(other);
    return otherCanvasImage && equals(*otherCanvasImage);
}

bool StyleCanvasImage::equals(const StyleCanvasImage& other) const
{
    return m_name == other.m_name;
}

Ref<CSSValue> StyleCanvasImage::computedStyleValue(const RenderStyle&) const
{
    return CSSCanvasValue::create(m_name);
}

bool StyleCanvasImage::isPending() const
{
    return false;
}

void StyleCanvasImage::load(CachedResourceLoader&, const ResourceLoaderOptions&)
{
}

RefPtr<Image> StyleCanvasImage::image(const RenderElement* renderer, const FloatSize&, bool) const
{
    if (!renderer)
        return &Image::nullImage();

    ASSERT(clients().contains(const_cast<RenderElement&>(*renderer)));
    RefPtr element = this->element(renderer->document());
    if (!element)
        return nullptr;
    return element->copiedImage();
}

bool StyleCanvasImage::knownToBeOpaque(const RenderElement&) const
{
    // FIXME: When CanvasRenderingContext2DSettings.alpha is implemented, this can be improved to check for it.
    return false;
}

FloatSize StyleCanvasImage::fixedSize(const RenderElement& renderer) const
{
    if (auto* element = this->element(renderer.document()))
        return FloatSize { element->size() };
    return { };
}

void StyleCanvasImage::didAddClient(RenderElement& renderer)
{
    if (auto* element = this->element(renderer.document()))
        InspectorInstrumentation::didChangeCSSCanvasClientNodes(*element);
}

void StyleCanvasImage::didRemoveClient(RenderElement& renderer)
{
    if (auto* element = this->element(renderer.document()))
        InspectorInstrumentation::didChangeCSSCanvasClientNodes(*element);
}

void StyleCanvasImage::canvasChanged(CanvasBase& canvasBase, const FloatRect& changedRect)
{
    ASSERT_UNUSED(canvasBase, is<HTMLCanvasElement>(canvasBase));
    ASSERT_UNUSED(canvasBase, m_element == &downcast<HTMLCanvasElement>(canvasBase));

    auto imageChangeRect = enclosingIntRect(changedRect);
    for (auto entry : clients()) {
        auto& client = entry.key;
        client.imageChanged(static_cast<WrappedImagePtr>(this), &imageChangeRect);
    }
}

void StyleCanvasImage::canvasResized(CanvasBase& canvasBase)
{
    ASSERT_UNUSED(canvasBase, is<HTMLCanvasElement>(canvasBase));
    ASSERT_UNUSED(canvasBase, m_element == &downcast<HTMLCanvasElement>(canvasBase));

    for (auto entry : clients()) {
        auto& client = entry.key;
        client.imageChanged(static_cast<WrappedImagePtr>(this));
    }
}

void StyleCanvasImage::canvasDestroyed(CanvasBase& canvasBase)
{
    ASSERT_UNUSED(canvasBase, is<HTMLCanvasElement>(canvasBase));
    ASSERT_UNUSED(canvasBase, m_element == &downcast<HTMLCanvasElement>(canvasBase));
    m_element = nullptr;
}

HTMLCanvasElement* StyleCanvasImage::element(Document& document) const
{
    if (!m_element) {
        m_element = document.getCSSCanvasElement(m_name);
        if (!m_element)
            return nullptr;
        m_element->addObserver(const_cast<StyleCanvasImage&>(*this));
    }
    return m_element.get();
}

} // namespace WebCore
