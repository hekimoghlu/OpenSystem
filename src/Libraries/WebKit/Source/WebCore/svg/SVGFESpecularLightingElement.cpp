/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
#include "SVGFESpecularLightingElement.h"

#include "FESpecularLighting.h"
#include "NodeName.h"
#include "RenderElement.h"
#include "RenderStyle.h"
#include "SVGFELightElement.h"
#include "SVGNames.h"
#include "SVGParserUtilities.h"
#include "SVGRenderStyle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFESpecularLightingElement);

inline SVGFESpecularLightingElement::SVGFESpecularLightingElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feSpecularLightingTag));
    
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFESpecularLightingElement::m_in1>();
        PropertyRegistry::registerProperty<SVGNames::specularConstantAttr, &SVGFESpecularLightingElement::m_specularConstant>();
        PropertyRegistry::registerProperty<SVGNames::specularExponentAttr, &SVGFESpecularLightingElement::m_specularExponent>();
        PropertyRegistry::registerProperty<SVGNames::surfaceScaleAttr, &SVGFESpecularLightingElement::m_surfaceScale>();
        PropertyRegistry::registerProperty<SVGNames::kernelUnitLengthAttr, &SVGFESpecularLightingElement::m_kernelUnitLengthX, &SVGFESpecularLightingElement::m_kernelUnitLengthY>();
    });
}

Ref<SVGFESpecularLightingElement> SVGFESpecularLightingElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFESpecularLightingElement(tagName, document));
}

void SVGFESpecularLightingElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::inAttr:
        Ref { m_in1 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::surfaceScaleAttr:
        Ref { m_surfaceScale }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::specularConstantAttr:
        Ref { m_specularConstant }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::specularExponentAttr:
        Ref { m_specularExponent }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::kernelUnitLengthAttr:
        if (auto result = parseNumberOptionalNumber(newValue)) {
            Ref { m_kernelUnitLengthX }->setBaseValInternal(result->first);
            Ref { m_kernelUnitLengthY }->setBaseValInternal(result->second);
        }
        break;
    default:
        break;
    }

    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

bool SVGFESpecularLightingElement::setFilterEffectAttribute(FilterEffect& filterEffect, const QualifiedName& attrName)
{
    auto& effect = downcast<FESpecularLighting>(filterEffect);
    auto lightElement = [this] {
        return SVGFELightElement::findLightElement(this);
    };

    switch (attrName.nodeName()) {
    case AttributeNames::lighting_colorAttr: {
        auto& style = renderer()->style();
        auto color = style.colorWithColorFilter(style.svgStyle().lightingColor());
        return effect.setLightingColor(color);
    }
    case AttributeNames::surfaceScaleAttr:
        return effect.setSurfaceScale(surfaceScale());
    case AttributeNames::specularConstantAttr:
        return effect.setSpecularConstant(specularConstant());
    case AttributeNames::specularExponentAttr:
        return effect.setSpecularExponent(specularExponent());
    case AttributeNames::azimuthAttr:
        return effect.lightSource()->setAzimuth(lightElement()->azimuth());
    case AttributeNames::elevationAttr:
        return effect.lightSource()->setElevation(lightElement()->elevation());
    case AttributeNames::xAttr:
        return effect.lightSource()->setX(lightElement()->x());
    case AttributeNames::yAttr:
        return effect.lightSource()->setY(lightElement()->y());
    case AttributeNames::zAttr:
        return effect.lightSource()->setZ(lightElement()->z());
    case AttributeNames::pointsAtXAttr:
        return effect.lightSource()->setPointsAtX(lightElement()->pointsAtX());
    case AttributeNames::pointsAtYAttr:
        return effect.lightSource()->setPointsAtY(lightElement()->pointsAtY());
    case AttributeNames::pointsAtZAttr:
        return effect.lightSource()->setPointsAtZ(lightElement()->pointsAtZ());
    case AttributeNames::limitingConeAngleAttr:
        return effect.lightSource()->setLimitingConeAngle(lightElement()->limitingConeAngle());
    default:
        break;
    }
    ASSERT_NOT_REACHED();
    return false;
}

void SVGFESpecularLightingElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        if (attrName == SVGNames::inAttr)
            updateSVGRendererForElementChange();
        else {
            ASSERT(attrName == SVGNames::specularConstantAttr || attrName == SVGNames::specularExponentAttr || attrName == SVGNames::surfaceScaleAttr || attrName == SVGNames::kernelUnitLengthAttr);
            primitiveAttributeChanged(attrName);
        }
        return;
    }

    SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
}

void SVGFESpecularLightingElement::lightElementAttributeChanged(const SVGFELightElement* lightElement, const QualifiedName& attrName)
{
    if (SVGFELightElement::findLightElement(this) != lightElement)
        return;

    // The light element has different attribute names so attrName can identify the requested attribute.
    primitiveAttributeChanged(attrName);
}

RefPtr<FilterEffect> SVGFESpecularLightingElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    RefPtr lightElement = SVGFELightElement::findLightElement(this);
    if (!lightElement)
        return nullptr;

    CheckedPtr renderer = this->renderer();
    if (!renderer)
        return nullptr;

    Ref lightSource = lightElement->lightSource();
    auto& style = renderer->style();

    auto color = style.colorWithColorFilter(style.svgStyle().lightingColor());

    return FESpecularLighting::create(color, surfaceScale(), specularConstant(), specularExponent(), kernelUnitLengthX(), kernelUnitLengthY(), WTFMove(lightSource));
}

} // namespace WebCore
