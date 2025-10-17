/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
#include "SVGFEDropShadowElement.h"

#include "NodeName.h"
#include "RenderElement.h"
#include "RenderStyle.h"
#include "SVGFilter.h"
#include "SVGNames.h"
#include "SVGParserUtilities.h"
#include "SVGRenderStyle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEDropShadowElement);

inline SVGFEDropShadowElement::SVGFEDropShadowElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feDropShadowTag));
    
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFEDropShadowElement::m_in1>();
        PropertyRegistry::registerProperty<SVGNames::dxAttr, &SVGFEDropShadowElement::m_dx>();
        PropertyRegistry::registerProperty<SVGNames::dyAttr, &SVGFEDropShadowElement::m_dy>();
        PropertyRegistry::registerProperty<SVGNames::stdDeviationAttr, &SVGFEDropShadowElement::m_stdDeviationX, &SVGFEDropShadowElement::m_stdDeviationY>();
    });
}

Ref<SVGFEDropShadowElement> SVGFEDropShadowElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEDropShadowElement(tagName, document));
}

void SVGFEDropShadowElement::setStdDeviation(float x, float y)
{
    Ref { m_stdDeviationX }->setBaseValInternal(x);
    Ref { m_stdDeviationY }->setBaseValInternal(y);
    updateSVGRendererForElementChange();
}

void SVGFEDropShadowElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::stdDeviationAttr:
        if (auto result = parseNumberOptionalNumber(newValue)) {
            Ref { m_stdDeviationX }->setBaseValInternal(result->first);
            Ref { m_stdDeviationY }->setBaseValInternal(result->second);
        }
        break;
    case AttributeNames::inAttr:
        Ref { m_in1 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::dxAttr:
        Ref { m_dx }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::dyAttr:
        Ref { m_dy }->setBaseValInternal(newValue.toFloat());
        break;
    default:
        break;
    }

    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGFEDropShadowElement::svgAttributeChanged(const QualifiedName& attrName)
{
    switch (attrName.nodeName()) {
    case AttributeNames::inAttr: {
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        break;
    }
    case AttributeNames::stdDeviationAttr: {
        if (stdDeviationX() < 0 || stdDeviationY() < 0) {
            InstanceInvalidationGuard guard(*this);
            markFilterEffectForRebuild();
            return;
        }
        FALLTHROUGH;
    }
    case AttributeNames::dxAttr:
    case AttributeNames::dyAttr: {
        InstanceInvalidationGuard guard(*this);
        primitiveAttributeChanged(attrName);
        break;
    }
    default:
        SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
        break;
    }
}

bool SVGFEDropShadowElement::setFilterEffectAttribute(FilterEffect& filterEffect, const QualifiedName& attrName)
{
    auto& effect = downcast<FEDropShadow>(filterEffect);
    switch (attrName.nodeName()) {
    case AttributeNames::stdDeviationAttr:
        return effect.setStdDeviationX(stdDeviationX()) || effect.setStdDeviationY(stdDeviationY());
    case AttributeNames::dxAttr:
        return effect.setDx(dx());
    case AttributeNames::dyAttr:
        return effect.setDy(dy());
    case AttributeNames::flood_colorAttr: {
        auto& style = renderer()->style();
        return effect.setShadowColor(style.colorResolvingCurrentColor(style.svgStyle().floodColor()));
    }
    case AttributeNames::flood_opacityAttr:
        return effect.setShadowOpacity(renderer()->style().svgStyle().floodOpacity());
    default:
        break;
    }
    ASSERT_NOT_REACHED();
    return false;
}

bool SVGFEDropShadowElement::isIdentity() const
{
    return !stdDeviationX() && !stdDeviationY() && !dx() && !dy();
}

IntOutsets SVGFEDropShadowElement::outsets(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits) const
{
    auto offset = SVGFilter::calculateResolvedSize({ dx(), dy() }, targetBoundingBox, primitiveUnits);
    auto stdDeviation = SVGFilter::calculateResolvedSize({ stdDeviationX(), stdDeviationY() }, targetBoundingBox, primitiveUnits);
    return FEDropShadow::calculateOutsets(offset, stdDeviation);
}

RefPtr<FilterEffect> SVGFEDropShadowElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    CheckedPtr renderer = this->renderer();
    if (!renderer)
        return nullptr;

    if (stdDeviationX() < 0 || stdDeviationY() < 0)
        return nullptr;

    auto& style = renderer->style();
    const SVGRenderStyle& svgStyle = style.svgStyle();
    
    Color color = style.colorWithColorFilter(svgStyle.floodColor());
    float opacity = svgStyle.floodOpacity();

    return FEDropShadow::create(stdDeviationX(), stdDeviationY(), dx(), dy(), color, opacity);
}

} // namespace WebCore
