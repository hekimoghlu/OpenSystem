/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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
#include "SVGFETurbulenceElement.h"

#include "NodeName.h"
#include "SVGNames.h"
#include "SVGParserUtilities.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFETurbulenceElement);

inline SVGFETurbulenceElement::SVGFETurbulenceElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feTurbulenceTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::baseFrequencyAttr, &SVGFETurbulenceElement::m_baseFrequencyX, &SVGFETurbulenceElement::m_baseFrequencyY>();
        PropertyRegistry::registerProperty<SVGNames::numOctavesAttr, &SVGFETurbulenceElement::m_numOctaves>();
        PropertyRegistry::registerProperty<SVGNames::seedAttr, &SVGFETurbulenceElement::m_seed>();
        PropertyRegistry::registerProperty<SVGNames::stitchTilesAttr, SVGStitchOptions, &SVGFETurbulenceElement::m_stitchTiles>();
        PropertyRegistry::registerProperty<SVGNames::typeAttr, TurbulenceType, &SVGFETurbulenceElement::m_type>();
    });
}

Ref<SVGFETurbulenceElement> SVGFETurbulenceElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFETurbulenceElement(tagName, document));
}

void SVGFETurbulenceElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::typeAttr: {
        TurbulenceType propertyValue = SVGPropertyTraits<TurbulenceType>::fromString(newValue);
        if (propertyValue != TurbulenceType::Unknown)
            Ref { m_type }->setBaseValInternal<TurbulenceType>(propertyValue);
        break;
    }
    case AttributeNames::stitchTilesAttr: {
        SVGStitchOptions propertyValue = SVGPropertyTraits<SVGStitchOptions>::fromString(newValue);
        if (propertyValue > 0)
            Ref { m_stitchTiles }->setBaseValInternal<SVGStitchOptions>(propertyValue);
        break;
    }
    case AttributeNames::baseFrequencyAttr:
        if (auto result = parseNumberOptionalNumber(newValue)) {
            Ref { m_baseFrequencyX }->setBaseValInternal(result->first);
            Ref { m_baseFrequencyY }->setBaseValInternal(result->second);
        }
        break;
    case AttributeNames::seedAttr:
        Ref { m_seed }->setBaseValInternal(newValue.toFloat());
        break;
    case AttributeNames::numOctavesAttr:
        Ref { m_numOctaves }->setBaseValInternal(parseInteger<unsigned>(newValue).value_or(0));
        break;
    default:
        break;
    }
    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

bool SVGFETurbulenceElement::setFilterEffectAttribute(FilterEffect& filterEffect, const QualifiedName& attrName)
{
    auto& effect = downcast<FETurbulence>(filterEffect);
    switch (attrName.nodeName()) {
    case AttributeNames::typeAttr:
        return effect.setType(type());
    case AttributeNames::stitchTilesAttr:
        return effect.setStitchTiles(stitchTiles());
    case AttributeNames::baseFrequencyAttr: {
        bool baseFrequencyXChanged = effect.setBaseFrequencyX(baseFrequencyX());
        bool baseFrequencyYChanged = effect.setBaseFrequencyY(baseFrequencyY());
        return baseFrequencyXChanged || baseFrequencyYChanged;
    }
    case AttributeNames::seedAttr:
        return effect.setSeed(seed());
    case AttributeNames::numOctavesAttr:
        return effect.setNumOctaves(numOctaves());
    default:
        break;
    }
    ASSERT_NOT_REACHED();
    return false;
}

void SVGFETurbulenceElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        primitiveAttributeChanged(attrName);
        return;
    }

    SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
}

RefPtr<FilterEffect> SVGFETurbulenceElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    if (baseFrequencyX() < 0 || baseFrequencyY() < 0)
        return nullptr;

    return FETurbulence::create(type(), baseFrequencyX(), baseFrequencyY(), numOctaves(), seed(), stitchTiles() == SVG_STITCHTYPE_STITCH);
}

} // namespace WebCore
