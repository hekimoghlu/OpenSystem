/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
#include "SVGFEMorphologyElement.h"

#include "FEMorphology.h"
#include "NodeName.h"
#include "SVGFilter.h"
#include "SVGNames.h"
#include "SVGParserUtilities.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEMorphologyElement);

inline SVGFEMorphologyElement::SVGFEMorphologyElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feMorphologyTag));
    
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFEMorphologyElement::m_in1>();
        PropertyRegistry::registerProperty<SVGNames::operatorAttr, MorphologyOperatorType, &SVGFEMorphologyElement::m_svgOperator>();
        PropertyRegistry::registerProperty<SVGNames::radiusAttr, &SVGFEMorphologyElement::m_radiusX, &SVGFEMorphologyElement::m_radiusY>();
    });
}

Ref<SVGFEMorphologyElement> SVGFEMorphologyElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEMorphologyElement(tagName, document));
}

void SVGFEMorphologyElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::operatorAttr: {
        MorphologyOperatorType propertyValue = SVGPropertyTraits<MorphologyOperatorType>::fromString(newValue);
        if (propertyValue != MorphologyOperatorType::Unknown)
            Ref { m_svgOperator }->setBaseValInternal<MorphologyOperatorType>(propertyValue);
        break;
    }
    case AttributeNames::inAttr:
        Ref { m_in1 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::radiusAttr:
        if (auto result = parseNumberOptionalNumber(newValue)) {
            Ref { m_radiusX }->setBaseValInternal(result->first);
            Ref { m_radiusY }->setBaseValInternal(result->second);
        }
        break;
    default:
        break;
    }

    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

bool SVGFEMorphologyElement::setFilterEffectAttribute(FilterEffect& effect, const QualifiedName& attrName)
{
    auto& feMorphology = downcast<FEMorphology>(effect);
    if (attrName == SVGNames::operatorAttr)
        return feMorphology.setMorphologyOperator(svgOperator());
    if (attrName == SVGNames::radiusAttr) {
        // Both setRadius functions should be evaluated separately.
        bool isRadiusXChanged = feMorphology.setRadiusX(radiusX());
        bool isRadiusYChanged = feMorphology.setRadiusY(radiusY());
        return isRadiusXChanged || isRadiusYChanged;
    }

    ASSERT_NOT_REACHED();
    return false;
}

void SVGFEMorphologyElement::svgAttributeChanged(const QualifiedName& attrName)
{
    switch (attrName.nodeName()) {
    case AttributeNames::inAttr: {
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        break;
    }
    case AttributeNames::operatorAttr:
    case AttributeNames::radiusAttr: {
        InstanceInvalidationGuard guard(*this);
        primitiveAttributeChanged(attrName);
        break;
    }
    default:
        SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
        break;
    }
}

bool SVGFEMorphologyElement::isIdentity() const
{
    return !std::max(0.0f, radiusX()) && !std::max(0.0f, radiusY());
}

IntOutsets SVGFEMorphologyElement::outsets(const FloatRect& targetBoundingBox, SVGUnitTypes::SVGUnitType primitiveUnits) const
{
    auto radius = SVGFilter::calculateResolvedSize({ radiusX(), radiusY() }, targetBoundingBox, primitiveUnits);
    return { static_cast<int>(radius.height()), static_cast<int>(radius.width()), static_cast<int>(radius.height()), static_cast<int>(radius.width()) };
}

RefPtr<FilterEffect> SVGFEMorphologyElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    // "A negative or zero value disables the effect of the given filter
    // primitive (i.e., the result is the filter input image)."
    // https://drafts.fxtf.org/filters/#element-attrdef-femorphology-radius
    // This is handled by FEMorphology.
    return FEMorphology::create(svgOperator(), radiusX(), radiusY());
}

} // namespace WebCore
