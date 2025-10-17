/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#include "SVGMarkerElement.h"

#include "LegacyRenderSVGResourceMarker.h"
#include "NodeName.h"
#include "RenderSVGResourceMarker.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGMarkerElement);

inline SVGMarkerElement::SVGMarkerElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGFitToViewBox(this)
{
    // Spec: If the markerWidth/markerHeight attribute is not specified, the effect is as if a value of "3" were specified.
    ASSERT(hasTagName(SVGNames::markerTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::refXAttr, &SVGMarkerElement::m_refX>();
        PropertyRegistry::registerProperty<SVGNames::refYAttr, &SVGMarkerElement::m_refY>();
        PropertyRegistry::registerProperty<SVGNames::markerWidthAttr, &SVGMarkerElement::m_markerWidth>();
        PropertyRegistry::registerProperty<SVGNames::markerHeightAttr, &SVGMarkerElement::m_markerHeight>();
        PropertyRegistry::registerProperty<SVGNames::markerUnitsAttr, SVGMarkerUnitsType, &SVGMarkerElement::m_markerUnits>();
        PropertyRegistry::registerProperty<SVGNames::orientAttr, &SVGMarkerElement::m_orientAngle, &SVGMarkerElement::m_orientType>();
    });
}

Ref<SVGMarkerElement> SVGMarkerElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGMarkerElement(tagName, document));
}

AffineTransform SVGMarkerElement::viewBoxToViewTransform(float viewWidth, float viewHeight) const
{
    return SVGFitToViewBox::viewBoxToViewTransform(viewBox(), preserveAspectRatio(), viewWidth, viewHeight);
}

void SVGMarkerElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGParsingError parseError = NoError;
    switch (name.nodeName()) {
    case AttributeNames::markerUnitsAttr: {
        auto propertyValue = SVGPropertyTraits<SVGMarkerUnitsType>::fromString(newValue);
        if (propertyValue > 0)
            Ref { m_markerUnits }->setBaseValInternal<SVGMarkerUnitsType>(propertyValue);
        return;
    }
    case AttributeNames::orientAttr: {
        auto pair = SVGPropertyTraits<std::pair<SVGAngleValue, SVGMarkerOrientType>>::fromString(newValue);
        Ref { m_orientAngle }->setBaseValInternal(pair.first);
        Ref { m_orientType }->setBaseValInternal(pair.second);
        return;
    }
    case AttributeNames::refXAttr:
        Ref { m_refX }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError));
        break;
    case AttributeNames::refYAttr:
        Ref { m_refY }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError));
        break;
    case AttributeNames::markerWidthAttr:
        Ref { m_markerWidth }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError));
        break;
    case AttributeNames::markerHeightAttr:
        Ref { m_markerHeight }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError));
        break;
    default:
        break;
    }
    reportAttributeParsingError(parseError, name, newValue);

    SVGFitToViewBox::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGMarkerElement::invalidateMarkerResource()
{
    if (document().settings().layerBasedSVGEngineEnabled()) {
        if (CheckedPtr markerRenderer = dynamicDowncast<RenderSVGResourceMarker>(renderer()))
            markerRenderer->invalidateMarker();
        return;
    }

    updateSVGRendererForElementChange();
}

void SVGMarkerElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        if (PropertyRegistry::isAnimatedLengthAttribute(attrName))
            updateRelativeLengthsInformation();
        // These properties affect the layout of the RenderSVGResourceMarker itself (due to viewBox + overflowClipRect handling).
        if (attrName == SVGNames::markerWidthAttr || attrName == SVGNames::markerHeightAttr)
            updateSVGRendererForElementChange();
        else
            invalidateMarkerResource();
        return;
    }
    
    if (SVGFitToViewBox::isKnownAttribute(attrName)) {
        updateSVGRendererForElementChange();
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

void SVGMarkerElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);

    if (change.source == ChildChange::Source::Parser)
        return;

    invalidateMarkerResource();
}

AtomString SVGMarkerElement::orient() const
{
    return getAttribute(SVGNames::orientAttr);
}

void SVGMarkerElement::setOrient(const AtomString& orient)
{
    setAttribute(SVGNames::orientAttr, orient);
}

void SVGMarkerElement::setOrientToAuto()
{
    Ref { m_orientType }->setBaseVal(SVGMarkerOrientAuto);
    invalidateMarkerResource();
}

void SVGMarkerElement::setOrientToAngle(const SVGAngle& angle)
{
    Ref { m_orientAngle }->baseVal()->newValueSpecifiedUnits(angle.unitType(), angle.valueInSpecifiedUnits());
    invalidateMarkerResource();
}

RenderPtr<RenderElement> SVGMarkerElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return createRenderer<RenderSVGResourceMarker>(*this, WTFMove(style));
    return createRenderer<LegacyRenderSVGResourceMarker>(*this, WTFMove(style));
}

bool SVGMarkerElement::selfHasRelativeLengths() const
{
    return refX().isRelative()
        || refY().isRelative()
        || markerWidth().isRelative()
        || markerHeight().isRelative();
}

}
