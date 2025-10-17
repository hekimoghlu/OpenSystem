/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#include "SVGFEDiffuseLightingElement.h"

#include "FEDiffuseLighting.h"
#include "NodeName.h"
#include "RenderElement.h"
#include "RenderStyle.h"
#include "SVGFELightElement.h"
#include "SVGNames.h"
#include "SVGParserUtilities.h"
#include "SVGRenderStyle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEDiffuseLightingElement);

inline SVGFEDiffuseLightingElement::SVGFEDiffuseLightingElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feDiffuseLightingTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFEDiffuseLightingElement::m_in1>();
        PropertyRegistry::registerProperty<SVGNames::diffuseConstantAttr, &SVGFEDiffuseLightingElement::m_diffuseConstant>();
        PropertyRegistry::registerProperty<SVGNames::surfaceScaleAttr, &SVGFEDiffuseLightingElement::m_surfaceScale>();
        PropertyRegistry::registerProperty<SVGNames::kernelUnitLengthAttr, &SVGFEDiffuseLightingElement::m_kernelUnitLengthX, &SVGFEDiffuseLightingElement::m_kernelUnitLengthY>();
    });
}

Ref<SVGFEDiffuseLightingElement> SVGFEDiffuseLightingElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEDiffuseLightingElement(tagName, document));
}

void SVGFEDiffuseLightingElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::inAttr:
        Ref { m_in1 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::surfaceScaleAttr:
        Ref { m_surfaceScale }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::diffuseConstantAttr:
        Ref { m_diffuseConstant }->setBaseValInternal(newValue.toFloat());
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

bool SVGFEDiffuseLightingElement::setFilterEffectAttribute(FilterEffect& filterEffect, const QualifiedName& attrName)
{
    auto& effect = downcast<FEDiffuseLighting>(filterEffect);
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
    case AttributeNames::diffuseConstantAttr:
        return effect.setDiffuseConstant(diffuseConstant());
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
    case AttributeNames::specularExponentAttr:
        return effect.lightSource()->setSpecularExponent(lightElement()->specularExponent());
    case AttributeNames::limitingConeAngleAttr:
        return effect.lightSource()->setLimitingConeAngle(lightElement()->limitingConeAngle());
    default:
        break;
    }
    ASSERT_NOT_REACHED();
    return false;
}

void SVGFEDiffuseLightingElement::svgAttributeChanged(const QualifiedName& attrName)
{
    switch (attrName.nodeName()) {
    case AttributeNames::inAttr: {
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        break;
    }
    case AttributeNames::diffuseConstantAttr:
    case AttributeNames::surfaceScaleAttr:
    case AttributeNames::kernelUnitLengthAttr: {
        InstanceInvalidationGuard guard(*this);
        primitiveAttributeChanged(attrName);
        break;
    }
    default:
        SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
        break;
    }
}

void SVGFEDiffuseLightingElement::lightElementAttributeChanged(const SVGFELightElement* lightElement, const QualifiedName& attrName)
{
    if (SVGFELightElement::findLightElement(this) != lightElement)
        return;

    // The light element has different attribute names.
    primitiveAttributeChanged(attrName);
}

RefPtr<FilterEffect> SVGFEDiffuseLightingElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    RefPtr lightElement = SVGFELightElement::findLightElement(this);
    if (!lightElement)
        return nullptr;

    CheckedPtr renderer = this->renderer();
    if (!renderer)
        return nullptr;

    Ref lightSource = lightElement->lightSource();
    auto& style = renderer->style();

    Color color = style.colorWithColorFilter(style.svgStyle().lightingColor());

    return FEDiffuseLighting::create(color, surfaceScale(), diffuseConstant(), kernelUnitLengthX(), kernelUnitLengthY(), WTFMove(lightSource));
}

} // namespace WebCore
