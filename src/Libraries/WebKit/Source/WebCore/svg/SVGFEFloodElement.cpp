/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#include "SVGFEFloodElement.h"

#include "FEFlood.h"
#include "RenderElement.h"
#include "RenderStyle.h"
#include "SVGNames.h"
#include "SVGRenderStyle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEFloodElement);

inline SVGFEFloodElement::SVGFEFloodElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feFloodTag));
}

Ref<SVGFEFloodElement> SVGFEFloodElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEFloodElement(tagName, document));
}

bool SVGFEFloodElement::setFilterEffectAttribute(FilterEffect& effect, const QualifiedName& attrName)
{
    CheckedPtr renderer = this->renderer();
    ASSERT(renderer);
    auto& style = renderer->style();

    auto& feFlood = downcast<FEFlood>(effect);
    if (attrName == SVGNames::flood_colorAttr)
        return feFlood.setFloodColor(style.colorResolvingCurrentColor(style.svgStyle().floodColor()));
    if (attrName == SVGNames::flood_opacityAttr)
        return feFlood.setFloodOpacity(style.svgStyle().floodOpacity());

    ASSERT_NOT_REACHED();
    return false;
}

RefPtr<FilterEffect> SVGFEFloodElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    CheckedPtr renderer = this->renderer();
    if (!renderer)
        return nullptr;

    auto& style = renderer->style();
    auto& svgStyle = style.svgStyle();

    auto color = style.colorWithColorFilter(svgStyle.floodColor());
    float opacity = svgStyle.floodOpacity();

    return FEFlood::create(color, opacity);
}

} // namespace WebCore
