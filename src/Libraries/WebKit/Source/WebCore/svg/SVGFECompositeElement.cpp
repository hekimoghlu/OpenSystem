/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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
#include "SVGFECompositeElement.h"

#include "FEComposite.h"
#include "NodeName.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFECompositeElement);

inline SVGFECompositeElement::SVGFECompositeElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feCompositeTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFECompositeElement::m_in1>();
        PropertyRegistry::registerProperty<SVGNames::in2Attr, &SVGFECompositeElement::m_in2>();
        PropertyRegistry::registerProperty<SVGNames::operatorAttr, CompositeOperationType, &SVGFECompositeElement::m_svgOperator>();
        PropertyRegistry::registerProperty<SVGNames::k1Attr, &SVGFECompositeElement::m_k1>();
        PropertyRegistry::registerProperty<SVGNames::k2Attr, &SVGFECompositeElement::m_k2>();
        PropertyRegistry::registerProperty<SVGNames::k3Attr, &SVGFECompositeElement::m_k3>();
        PropertyRegistry::registerProperty<SVGNames::k4Attr, &SVGFECompositeElement::m_k4>();
    });
}

Ref<SVGFECompositeElement> SVGFECompositeElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFECompositeElement(tagName, document));
}

void SVGFECompositeElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::operatorAttr: {
        CompositeOperationType propertyValue = SVGPropertyTraits<CompositeOperationType>::fromString(newValue);
        if (enumToUnderlyingType(propertyValue))
            Ref { m_svgOperator }->setBaseValInternal<CompositeOperationType>(propertyValue);
        break;
    }
    case AttributeNames::inAttr:
        Ref { m_in1 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::in2Attr:
        Ref { m_in2 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::k1Attr:
        Ref { m_k1 }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::k2Attr:
        Ref { m_k2 }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::k3Attr:
        Ref { m_k3 }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::k4Attr:
        Ref { m_k4 }->setBaseValInternal(newValue.toFloat());
        break;
    default:
        break;
    }

    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

bool SVGFECompositeElement::setFilterEffectAttribute(FilterEffect& filterEffect, const QualifiedName& attrName)
{
    auto& effect = downcast<FEComposite>(filterEffect);
    switch (attrName.nodeName()) {
    case AttributeNames::operatorAttr:
        return effect.setOperation(svgOperator());
    case AttributeNames::k1Attr:
        return effect.setK1(k1());
    case AttributeNames::k2Attr:
        return effect.setK2(k2());
    case AttributeNames::k3Attr:
        return effect.setK3(k3());
    case AttributeNames::k4Attr:
        return effect.setK4(k4());
    default:
        break;
    }
    ASSERT_NOT_REACHED();
    return false;
}


void SVGFECompositeElement::svgAttributeChanged(const QualifiedName& attrName)
{
    switch (attrName.nodeName()) {
    case AttributeNames::inAttr:
    case AttributeNames::in2Attr: {
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        break;
    }
    case AttributeNames::k1Attr:
    case AttributeNames::k2Attr:
    case AttributeNames::k3Attr:
    case AttributeNames::k4Attr:
    case AttributeNames::operatorAttr: {
        InstanceInvalidationGuard guard(*this);
        primitiveAttributeChanged(attrName);
        break;
    }
    default:
        SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
        break;
    }
}

RefPtr<FilterEffect> SVGFECompositeElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    return FEComposite::create(svgOperator(), k1(), k2(), k3(), k4());
}

} // namespace WebCore
