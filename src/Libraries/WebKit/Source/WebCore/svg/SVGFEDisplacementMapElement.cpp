/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
#include "SVGFEDisplacementMapElement.h"

#include "FEDisplacementMap.h"
#include "NodeName.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEDisplacementMapElement);

inline SVGFEDisplacementMapElement::SVGFEDisplacementMapElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feDisplacementMapTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFEDisplacementMapElement::m_in1>();
        PropertyRegistry::registerProperty<SVGNames::in2Attr, &SVGFEDisplacementMapElement::m_in2>();
        PropertyRegistry::registerProperty<SVGNames::xChannelSelectorAttr, ChannelSelectorType, &SVGFEDisplacementMapElement::m_xChannelSelector>();
        PropertyRegistry::registerProperty<SVGNames::yChannelSelectorAttr, ChannelSelectorType, &SVGFEDisplacementMapElement::m_yChannelSelector>();
        PropertyRegistry::registerProperty<SVGNames::scaleAttr, &SVGFEDisplacementMapElement::m_scale>();
    });
}

Ref<SVGFEDisplacementMapElement> SVGFEDisplacementMapElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEDisplacementMapElement(tagName, document));
}

void SVGFEDisplacementMapElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::xChannelSelectorAttr: {
        auto propertyValue = SVGPropertyTraits<ChannelSelectorType>::fromString(newValue);
        if (enumToUnderlyingType(propertyValue))
            Ref { m_xChannelSelector }->setBaseValInternal<ChannelSelectorType>(propertyValue);
        break;
    }
    case AttributeNames::yChannelSelectorAttr: {
        auto propertyValue = SVGPropertyTraits<ChannelSelectorType>::fromString(newValue);
        if (enumToUnderlyingType(propertyValue))
            Ref { m_yChannelSelector }->setBaseValInternal<ChannelSelectorType>(propertyValue);
        break;
    }
    case AttributeNames::inAttr:
        Ref { m_in1 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::in2Attr:
        Ref { m_in2 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::scaleAttr:
        Ref { m_scale }->setBaseValInternal(newValue.toFloat());
        break;
    default:
        break;
    }

    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

bool SVGFEDisplacementMapElement::setFilterEffectAttribute(FilterEffect& filterEffect, const QualifiedName& attrName)
{
    auto& effect = downcast<FEDisplacementMap>(filterEffect);
    switch (attrName.nodeName()) {
    case AttributeNames::xChannelSelectorAttr:
        return effect.setXChannelSelector(xChannelSelector());
    case AttributeNames::yChannelSelectorAttr:
        return effect.setYChannelSelector(yChannelSelector());
    case AttributeNames::scaleAttr:
        return effect.setScale(scale());
    default:
        break;
    }
    ASSERT_NOT_REACHED();
    return false;
}

void SVGFEDisplacementMapElement::svgAttributeChanged(const QualifiedName& attrName)
{
    switch (attrName.nodeName()) {
    case AttributeNames::inAttr:
    case AttributeNames::in2Attr: {
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        break;
    }
    case AttributeNames::xChannelSelectorAttr:
    case AttributeNames::yChannelSelectorAttr:
    case AttributeNames::scaleAttr: {
        InstanceInvalidationGuard guard(*this);
        primitiveAttributeChanged(attrName);
        break;
    }
    default:
        SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
        break;
    }
}

RefPtr<FilterEffect> SVGFEDisplacementMapElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    return FEDisplacementMap::create(xChannelSelector(), yChannelSelector(), scale());
}

} // namespace WebCore
