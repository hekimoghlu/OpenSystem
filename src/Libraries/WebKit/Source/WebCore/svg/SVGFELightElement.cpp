/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
#include "SVGFELightElement.h"

#include "ElementChildIteratorInlines.h"
#include "LegacyRenderSVGResource.h"
#include "NodeName.h"
#include "RenderObject.h"
#include "SVGElementTypeHelpers.h"
#include "SVGFEDiffuseLightingElement.h"
#include "SVGFEDistantLightElement.h"
#include "SVGFEPointLightElement.h"
#include "SVGFESpecularLightingElement.h"
#include "SVGFESpotLightElement.h"
#include "SVGFilterElement.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFELightElement);

SVGFELightElement::SVGFELightElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::azimuthAttr, &SVGFELightElement::m_azimuth>();
        PropertyRegistry::registerProperty<SVGNames::elevationAttr, &SVGFELightElement::m_elevation>();
        PropertyRegistry::registerProperty<SVGNames::xAttr, &SVGFELightElement::m_x>();
        PropertyRegistry::registerProperty<SVGNames::yAttr, &SVGFELightElement::m_y>();
        PropertyRegistry::registerProperty<SVGNames::zAttr, &SVGFELightElement::m_z>();
        PropertyRegistry::registerProperty<SVGNames::pointsAtXAttr, &SVGFELightElement::m_pointsAtX>();
        PropertyRegistry::registerProperty<SVGNames::pointsAtYAttr, &SVGFELightElement::m_pointsAtY>();
        PropertyRegistry::registerProperty<SVGNames::pointsAtZAttr, &SVGFELightElement::m_pointsAtZ>();
        PropertyRegistry::registerProperty<SVGNames::specularExponentAttr, &SVGFELightElement::m_specularExponent>();
        PropertyRegistry::registerProperty<SVGNames::limitingConeAngleAttr, &SVGFELightElement::m_limitingConeAngle>();
    });
}

SVGFELightElement* SVGFELightElement::findLightElement(const SVGElement* svgElement)
{
    for (auto& child : childrenOfType<SVGElement>(*svgElement)) {
        if (is<SVGFEDistantLightElement>(child) || is<SVGFEPointLightElement>(child) || is<SVGFESpotLightElement>(child))
            return static_cast<SVGFELightElement*>(const_cast<SVGElement*>(&child));
    }
    return nullptr;
}

void SVGFELightElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::azimuthAttr:
        Ref { m_azimuth }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::elevationAttr:
        Ref { m_elevation }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::xAttr:
        Ref { m_x }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::yAttr:
        Ref { m_y }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::zAttr:
        Ref { m_z }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::pointsAtXAttr:
        Ref { m_pointsAtX }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::pointsAtYAttr:
        Ref { m_pointsAtY }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::pointsAtZAttr:
        Ref { m_pointsAtZ }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::specularExponentAttr:
        Ref { m_specularExponent }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::limitingConeAngleAttr:
        Ref { m_limitingConeAngle }->setBaseValInternal(newValue.toFloat());
        break;
    default:
        break;
    }

    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGFELightElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        ASSERT(attrName == SVGNames::azimuthAttr || attrName == SVGNames::elevationAttr || attrName == SVGNames::xAttr || attrName == SVGNames::yAttr
            || attrName == SVGNames::zAttr || attrName == SVGNames::pointsAtXAttr || attrName == SVGNames::pointsAtYAttr || attrName == SVGNames::pointsAtZAttr
            || attrName == SVGNames::specularExponentAttr || attrName == SVGNames::limitingConeAngleAttr);

        RefPtr parent = parentElement();
        if (!parent)
            return;

        CheckedPtr renderer = parent->renderer();
        if (!renderer || !renderer->isRenderOrLegacyRenderSVGResourceFilterPrimitive())
            return;

        if (auto* lightingElement = dynamicDowncast<SVGFEDiffuseLightingElement>(*parent)) {
            InstanceInvalidationGuard guard(*this);
            lightingElement->lightElementAttributeChanged(this, attrName);
        } else if (auto* lightingElement = dynamicDowncast<SVGFESpecularLightingElement>(*parent)) {
            InstanceInvalidationGuard guard(*this);
            lightingElement->lightElementAttributeChanged(this, attrName);
        }

        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

void SVGFELightElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);

    if (change.source == ChildChange::Source::Parser)
        return;

    SVGFilterPrimitiveStandardAttributes::invalidateFilterPrimitiveParent(this);
}

}
