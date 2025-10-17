/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#include "SVGFEOffsetElement.h"

#include "FEOffset.h"
#include "NodeName.h"
#include "SVGFilter.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEOffsetElement);

inline SVGFEOffsetElement::SVGFEOffsetElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feOffsetTag));
    
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFEOffsetElement::m_in1>();
        PropertyRegistry::registerProperty<SVGNames::dxAttr, &SVGFEOffsetElement::m_dx>();
        PropertyRegistry::registerProperty<SVGNames::dyAttr, &SVGFEOffsetElement::m_dy>();
    });
}

Ref<SVGFEOffsetElement> SVGFEOffsetElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEOffsetElement(tagName, document));
}

void SVGFEOffsetElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::dxAttr:
        Ref { m_dx }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::dyAttr:
        Ref { m_dy }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::inAttr:
        Ref { m_in1 }->setBaseValInternal(newValue);
        break;
    default:
        break;
    }

    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGFEOffsetElement::svgAttributeChanged(const QualifiedName& attrName)
{
    switch (attrName.nodeName()) {
    case AttributeNames::inAttr: {
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        break;
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

bool SVGFEOffsetElement::setFilterEffectAttribute(FilterEffect& effect, const QualifiedName& attrName)
{
    auto& offset = downcast<FEOffset>(effect);

    if (attrName == SVGNames::dxAttr)
        return offset.setDx(dx());
    if (attrName == SVGNames::dyAttr)
        return offset.setDy(dy());

    ASSERT_NOT_REACHED();
    return false;
}

bool SVGFEOffsetElement::isIdentity() const
{
    return !dx() && !dy();
}

IntOutsets SVGFEOffsetElement::outsets(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits) const
{
    auto offset = SVGFilter::calculateResolvedSize({ dx(), dy() }, targetBoundingBox, primitiveUnits);
    return FEOffset::calculateOutsets(offset);
}

RefPtr<FilterEffect> SVGFEOffsetElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    return FEOffset::create(dx(), dy());
}

} // namespace WebCore
