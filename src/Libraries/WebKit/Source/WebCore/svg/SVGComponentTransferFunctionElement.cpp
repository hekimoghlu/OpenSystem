/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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
#include "SVGComponentTransferFunctionElement.h"

#include "NodeName.h"
#include "SVGComponentTransferFunctionElementInlines.h"
#include "SVGFEComponentTransferElement.h"
#include "SVGNames.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGComponentTransferFunctionElement);

SVGComponentTransferFunctionElement::SVGComponentTransferFunctionElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::typeAttr, ComponentTransferType, &SVGComponentTransferFunctionElement::m_type>();
        PropertyRegistry::registerProperty<SVGNames::tableValuesAttr, &SVGComponentTransferFunctionElement::m_tableValues>();
        PropertyRegistry::registerProperty<SVGNames::slopeAttr, &SVGComponentTransferFunctionElement::m_slope>();
        PropertyRegistry::registerProperty<SVGNames::interceptAttr, &SVGComponentTransferFunctionElement::m_intercept>();
        PropertyRegistry::registerProperty<SVGNames::amplitudeAttr, &SVGComponentTransferFunctionElement::m_amplitude>();
        PropertyRegistry::registerProperty<SVGNames::exponentAttr, &SVGComponentTransferFunctionElement::m_exponent>();
        PropertyRegistry::registerProperty<SVGNames::offsetAttr, &SVGComponentTransferFunctionElement::m_offset>();
    });
}

void SVGComponentTransferFunctionElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::typeAttr: {
        ComponentTransferType propertyValue = SVGPropertyTraits<ComponentTransferType>::fromString(newValue);
        if (enumToUnderlyingType(propertyValue))
            m_type->setBaseValInternal<ComponentTransferType>(propertyValue);
        break;
    }
    case AttributeNames::tableValuesAttr:
        m_tableValues->baseVal()->parse(newValue);
        break;
    case AttributeNames::slopeAttr:
        m_slope->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::interceptAttr:
        m_intercept->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::amplitudeAttr:
        m_amplitude->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::exponentAttr:
        m_exponent->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::offsetAttr:
        m_offset->setBaseValInternal(newValue.toFloat());
        break;
    default:
        break;
    }

    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGComponentTransferFunctionElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        if (RefPtr transferElement = dynamicDowncast<SVGFEComponentTransferElement>(parentElement())) {
            InstanceInvalidationGuard guard(*this);
            transferElement->transferFunctionAttributeChanged(*this, attrName);
        }
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

ComponentTransferFunction SVGComponentTransferFunctionElement::transferFunction() const
{
    return {
        type(),
        slope(),
        intercept(),
        amplitude(),
        exponent(),
        offset(),
        tableValues()
    };
}

}
