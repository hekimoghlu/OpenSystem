/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#include "SVGGradientElement.h"

#include "ElementChildIteratorInlines.h"
#include "LegacyRenderSVGResourceLinearGradient.h"
#include "LegacyRenderSVGResourceRadialGradient.h"
#include "NodeName.h"
#include "RenderSVGResourceGradient.h"
#include "SVGElementTypeHelpers.h"
#include "SVGStopElement.h"
#include "SVGTransformable.h"
#include "StyleResolver.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGGradientElement);

SVGGradientElement::SVGGradientElement(const QualifiedName& tagName, Document& document, UniqueRef<SVGPropertyRegistry>&& propertyRegistry)
    : SVGElement(tagName, document, WTFMove(propertyRegistry))
    , SVGURIReference(this)
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::spreadMethodAttr, SVGSpreadMethodType, &SVGGradientElement::m_spreadMethod>();
        PropertyRegistry::registerProperty<SVGNames::gradientUnitsAttr, SVGUnitTypes::SVGUnitType, &SVGGradientElement::m_gradientUnits>();
        PropertyRegistry::registerProperty<SVGNames::gradientTransformAttr, &SVGGradientElement::m_gradientTransform>();
    });
}

void SVGGradientElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::gradientUnitsAttr: {
        auto propertyValue = SVGPropertyTraits<SVGUnitTypes::SVGUnitType>::fromString(newValue);
        if (propertyValue > 0)
            Ref { m_gradientUnits }->setBaseValInternal<SVGUnitTypes::SVGUnitType>(propertyValue);
        break;
    }
    case AttributeNames::gradientTransformAttr:
        Ref { m_gradientTransform }->baseVal()->parse(newValue);
        break;
    case AttributeNames::spreadMethodAttr: {
        auto propertyValue = SVGPropertyTraits<SVGSpreadMethodType>::fromString(newValue);
        if (propertyValue > 0)
            Ref { m_spreadMethod }->setBaseValInternal<SVGSpreadMethodType>(propertyValue);
        break;
    }
    default:
        break;
    }

    SVGURIReference::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGGradientElement::invalidateGradientResource()
{
    if (document().settings().layerBasedSVGEngineEnabled()) {
        if (CheckedPtr gradientRenderer = dynamicDowncast<RenderSVGResourceGradient>(renderer()))
            gradientRenderer->invalidateGradient();
        return;
    }

    updateSVGRendererForElementChange();
}

void SVGGradientElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName) || SVGURIReference::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        invalidateGradientResource();
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}
    
void SVGGradientElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);
    if (change.source == ChildChange::Source::Parser)
        return;

    invalidateGradientResource();
}

GradientColorStops SVGGradientElement::buildStops()
{
    GradientColorStops stops;
    float previousOffset = 0.0f;
    for (auto& stop : childrenOfType<SVGStopElement>(*this)) {
        auto monotonicallyIncreasingOffset = std::clamp(stop.offset(), previousOffset, 1.0f);
        previousOffset = monotonicallyIncreasingOffset;
        stops.addColorStop({ monotonicallyIncreasingOffset, stop.stopColorIncludingOpacity() });
    }
    return stops;
}

}
