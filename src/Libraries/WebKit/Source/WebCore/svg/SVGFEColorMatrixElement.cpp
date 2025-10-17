/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#include "SVGFEColorMatrixElement.h"

#include "FEColorMatrix.h"
#include "NodeName.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEColorMatrixElement);

inline SVGFEColorMatrixElement::SVGFEColorMatrixElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feColorMatrixTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFEColorMatrixElement::m_in1>();
        PropertyRegistry::registerProperty<SVGNames::typeAttr, ColorMatrixType, &SVGFEColorMatrixElement::m_type>();
        PropertyRegistry::registerProperty<SVGNames::valuesAttr, &SVGFEColorMatrixElement::m_values>();
    });
}

Ref<SVGFEColorMatrixElement> SVGFEColorMatrixElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEColorMatrixElement(tagName, document));
}

bool SVGFEColorMatrixElement::isInvalidValuesLength() const
{
    auto filterType = type();
    auto size = values().size();

    return (filterType == ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX    && size != 20)
        || (filterType == ColorMatrixType::FECOLORMATRIX_TYPE_HUEROTATE && size != 1)
        || (filterType == ColorMatrixType::FECOLORMATRIX_TYPE_SATURATE  && size != 1);
}

void SVGFEColorMatrixElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::typeAttr: {
        auto propertyValue = SVGPropertyTraits<ColorMatrixType>::fromString(newValue);
        if (enumToUnderlyingType(propertyValue))
            Ref { m_type }->setBaseValInternal<ColorMatrixType>(propertyValue);
        break;
    }
    case AttributeNames::inAttr:
        Ref { m_in1 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::valuesAttr:
        Ref { m_values }->baseVal()->parse(newValue);
        break;
    default:
        break;
    }

    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

bool SVGFEColorMatrixElement::setFilterEffectAttribute(FilterEffect& effect, const QualifiedName& attrName)
{
    auto& feColorMatrix = downcast<FEColorMatrix>(effect);

    if (attrName == SVGNames::typeAttr)
        return feColorMatrix.setType(type());
    if (attrName == SVGNames::valuesAttr)
        return feColorMatrix.setValues(values());

    ASSERT_NOT_REACHED();
    return false;
}

void SVGFEColorMatrixElement::svgAttributeChanged(const QualifiedName& attrName)
{
    switch (attrName.nodeName()) {
    case AttributeNames::inAttr: {
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        break;
    }
    case AttributeNames::typeAttr:
    case AttributeNames::valuesAttr: {
        InstanceInvalidationGuard guard(*this);
        if (isInvalidValuesLength())
            markFilterEffectForRebuild();
        else
            primitiveAttributeChanged(attrName);
        break;
    }
    default:
        SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
        break;
    }
}

RefPtr<FilterEffect> SVGFEColorMatrixElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    Vector<float> filterValues;
    ColorMatrixType filterType = type();

    // Use defaults if values is empty (SVG 1.1 15.10).
    if (!hasAttribute(SVGNames::valuesAttr)) {
        switch (filterType) {
        case ColorMatrixType::FECOLORMATRIX_TYPE_MATRIX: {
            static constexpr unsigned matrixValueCount = 20;
            filterValues = Vector<float>(matrixValueCount, [](size_t i) {
                return (i % 6) ? 0.0 : 1.0;
            });
            break;
        }
        case ColorMatrixType::FECOLORMATRIX_TYPE_HUEROTATE:
            filterValues = { 0 };
            break;
        case ColorMatrixType::FECOLORMATRIX_TYPE_SATURATE:
            filterValues = { 1 };
            break;
        default:
            break;
        }
    } else {
        if (isInvalidValuesLength())
            return nullptr;

        filterValues = values();
        filterValues.shrinkToFit();
    }

    return FEColorMatrix::create(filterType, WTFMove(filterValues));
}

} // namespace WebCore
