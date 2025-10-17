/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#include "SVGFEComponentTransferElement.h"

#include "ElementChildIteratorInlines.h"
#include "FEComponentTransfer.h"
#include "NodeName.h"
#include "SVGComponentTransferFunctionElement.h"
#include "SVGComponentTransferFunctionElementInlines.h"
#include "SVGElementTypeHelpers.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEComponentTransferElement);

inline SVGFEComponentTransferElement::SVGFEComponentTransferElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feComponentTransferTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFEComponentTransferElement::m_in1>();
    });
}

Ref<SVGFEComponentTransferElement> SVGFEComponentTransferElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEComponentTransferElement(tagName, document));
}

void SVGFEComponentTransferElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::inAttr)
        Ref { m_in1 }->setBaseValInternal(newValue);

    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGFEComponentTransferElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (attrName == SVGNames::inAttr) {
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        return;
    }

    SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
}

RefPtr<FilterEffect> SVGFEComponentTransferElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    ComponentTransferFunctions functions;

    for (auto& child : childrenOfType<SVGComponentTransferFunctionElement>(*this))
        functions[child.channel()] = child.transferFunction();

    return FEComponentTransfer::create(WTFMove(functions));
}

static bool isRelevantTransferFunctionElement(const Element& child)
{
    auto name = child.elementName();

    ASSERT(is<SVGComponentTransferFunctionElement>(child));

    for (auto laterSibling = child.nextElementSibling(); laterSibling; laterSibling = laterSibling->nextElementSibling()) {
        if (laterSibling->elementName() == name)
            return false;
    }

    return true;
}

bool SVGFEComponentTransferElement::setFilterEffectAttributeFromChild(FilterEffect& filterEffect, const Element& childElement, const QualifiedName& attrName)
{
    ASSERT(isRelevantTransferFunctionElement(childElement));

    RefPtr child = dynamicDowncast<SVGComponentTransferFunctionElement>(childElement);
    ASSERT(child);
    if (!child)
        return false;

    auto& effect = downcast<FEComponentTransfer>(filterEffect);

    switch (attrName.nodeName()) {
    case AttributeNames::typeAttr:
        return effect.setType(child->channel(), child->type());
    case AttributeNames::slopeAttr:
        return effect.setSlope(child->channel(), child->slope());
    case AttributeNames::interceptAttr:
        return effect.setIntercept(child->channel(), child->intercept());
    case AttributeNames::amplitudeAttr:
        return effect.setAmplitude(child->channel(), child->amplitude());
    case AttributeNames::exponentAttr:
        return effect.setExponent(child->channel(), child->exponent());
    case AttributeNames::offsetAttr:
        return effect.setOffset(child->channel(), child->offset());
    case AttributeNames::tableValuesAttr:
        return effect.setTableValues(child->channel(), child->tableValues());
    default:
        break;
    }
    return false;
}

void SVGFEComponentTransferElement::transferFunctionAttributeChanged(SVGComponentTransferFunctionElement& child, const QualifiedName& attrName)
{
    ASSERT(child.parentNode() == this);

    if (!isRelevantTransferFunctionElement(child))
        return;

    primitiveAttributeOnChildChanged(child, attrName);
}

} // namespace WebCore
