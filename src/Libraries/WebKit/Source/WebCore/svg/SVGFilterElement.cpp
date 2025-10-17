/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
#include "SVGFilterElement.h"

#include "LegacyRenderSVGResourceFilter.h"
#include "NodeName.h"
#include "RenderSVGResourceFilter.h"
#include "SVGElementInlines.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include "SVGNames.h"
#include "SVGParserUtilities.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFilterElement);

inline SVGFilterElement::SVGFilterElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGURIReference(this)
{
    // Spec: If the x/y attribute is not specified, the effect is as if a value of "-10%" were specified.
    // Spec: If the width/height attribute is not specified, the effect is as if a value of "120%" were specified.
    ASSERT(hasTagName(SVGNames::filterTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::filterUnitsAttr, SVGUnitTypes::SVGUnitType, &SVGFilterElement::m_filterUnits>();
        PropertyRegistry::registerProperty<SVGNames::primitiveUnitsAttr, SVGUnitTypes::SVGUnitType, &SVGFilterElement::m_primitiveUnits>();
        PropertyRegistry::registerProperty<SVGNames::xAttr, &SVGFilterElement::m_x>();
        PropertyRegistry::registerProperty<SVGNames::yAttr, &SVGFilterElement::m_y>();
        PropertyRegistry::registerProperty<SVGNames::widthAttr, &SVGFilterElement::m_width>();
        PropertyRegistry::registerProperty<SVGNames::heightAttr, &SVGFilterElement::m_height>();
    });
}

Ref<SVGFilterElement> SVGFilterElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFilterElement(tagName, document));
}

void SVGFilterElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGParsingError parseError = NoError;

    switch (name.nodeName()) {
    case AttributeNames::filterUnitsAttr: {
        SVGUnitTypes::SVGUnitType propertyValue = SVGPropertyTraits<SVGUnitTypes::SVGUnitType>::fromString(newValue);
        if (propertyValue > 0)
            Ref { m_filterUnits }->setBaseValInternal<SVGUnitTypes::SVGUnitType>(propertyValue);
        break;
    }
    case AttributeNames::primitiveUnitsAttr: {
        SVGUnitTypes::SVGUnitType propertyValue = SVGPropertyTraits<SVGUnitTypes::SVGUnitType>::fromString(newValue);
        if (propertyValue > 0)
            Ref { m_primitiveUnits }->setBaseValInternal<SVGUnitTypes::SVGUnitType>(propertyValue);
        break;
    }
    case AttributeNames::xAttr:
        Ref { m_x }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError));
        break;
    case AttributeNames::yAttr:
        Ref { m_y }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError));
        break;
    case AttributeNames::widthAttr:
        Ref { m_width }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError));
        break;
    case AttributeNames::heightAttr:
        Ref { m_height }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError));
        break;
    default:
        break;
    }
    reportAttributeParsingError(parseError, name, newValue);

    SVGURIReference::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGFilterElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isAnimatedLengthAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        setPresentationalHintStyleIsDirty();
        return;
    }

    if (PropertyRegistry::isKnownAttribute(attrName) || SVGURIReference::isKnownAttribute(attrName)) {
        updateSVGRendererForElementChange();
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

void SVGFilterElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);

    if (change.source == ChildChange::Source::Parser)
        return;

    if (document().settings().layerBasedSVGEngineEnabled()) {
        if (CheckedPtr filterRenderer = dynamicDowncast<RenderSVGResourceFilter>(renderer()))
            filterRenderer->invalidateFilter();
        return;
    }

    updateSVGRendererForElementChange();
}

RenderPtr<RenderElement> SVGFilterElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return createRenderer<RenderSVGResourceFilter>(*this, WTFMove(style));

    return createRenderer<LegacyRenderSVGResourceFilter>(*this, WTFMove(style));
}

bool SVGFilterElement::childShouldCreateRenderer(const Node& child) const
{
    using namespace ElementNames;

    auto* childElement = dynamicDowncast<SVGElement>(child);
    if (!childElement)
        return false;

    switch (childElement->elementName()) {
    case SVG::feBlend:
    case SVG::feColorMatrix:
    case SVG::feComponentTransfer:
    case SVG::feComposite:
    case SVG::feConvolveMatrix:
    case SVG::feDiffuseLighting:
    case SVG::feDisplacementMap:
    case SVG::feDistantLight:
    case SVG::feDropShadow:
    case SVG::feFlood:
    case SVG::feFuncA:
    case SVG::feFuncB:
    case SVG::feFuncG:
    case SVG::feFuncR:
    case SVG::feGaussianBlur:
    case SVG::feImage:
    case SVG::feMerge:
    case SVG::feMergeNode:
    case SVG::feMorphology:
    case SVG::feOffset:
    case SVG::fePointLight:
    case SVG::feSpecularLighting:
    case SVG::feSpotLight:
    case SVG::feTile:
    case SVG::feTurbulence:
        return true;
    default:
        break;
    }

    return false;
}

}
