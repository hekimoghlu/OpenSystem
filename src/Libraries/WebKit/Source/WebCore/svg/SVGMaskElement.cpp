/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#include "SVGMaskElement.h"

#include "LegacyRenderSVGResourceMaskerInlines.h"
#include "NodeName.h"
#include "RenderElementInlines.h"
#include "RenderSVGResourceMasker.h"
#include "SVGElementInlines.h"
#include "SVGLayerTransformComputation.h"
#include "SVGNames.h"
#include "SVGRenderSupport.h"
#include "SVGStringList.h"
#include "SVGUnitTypes.h"
#include "StyleResolver.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGMaskElement);

inline SVGMaskElement::SVGMaskElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGTests(this)
{
    // Spec: If the x/y attribute is not specified, the effect is as if a value of "-10%" were specified.
    // Spec: If the width/height attribute is not specified, the effect is as if a value of "120%" were specified.
    ASSERT(hasTagName(SVGNames::maskTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::xAttr, &SVGMaskElement::m_x>();
        PropertyRegistry::registerProperty<SVGNames::yAttr, &SVGMaskElement::m_y>();
        PropertyRegistry::registerProperty<SVGNames::widthAttr, &SVGMaskElement::m_width>();
        PropertyRegistry::registerProperty<SVGNames::heightAttr, &SVGMaskElement::m_height>();
        PropertyRegistry::registerProperty<SVGNames::maskUnitsAttr, SVGUnitTypes::SVGUnitType, &SVGMaskElement::m_maskUnits>();
        PropertyRegistry::registerProperty<SVGNames::maskContentUnitsAttr, SVGUnitTypes::SVGUnitType, &SVGMaskElement::m_maskContentUnits>();
    });
}

Ref<SVGMaskElement> SVGMaskElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGMaskElement(tagName, document));
}

void SVGMaskElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGParsingError parseError = NoError;
    switch (name.nodeName()) {
    case AttributeNames::maskUnitsAttr: {
        auto propertyValue = SVGPropertyTraits<SVGUnitTypes::SVGUnitType>::fromString(newValue);
        if (propertyValue > 0)
            Ref { m_maskUnits }->setBaseValInternal<SVGUnitTypes::SVGUnitType>(propertyValue);
        break;
    }
    case AttributeNames::maskContentUnitsAttr: {
        auto propertyValue = SVGPropertyTraits<SVGUnitTypes::SVGUnitType>::fromString(newValue);
        if (propertyValue > 0)
            Ref { m_maskContentUnits }->setBaseValInternal<SVGUnitTypes::SVGUnitType>(propertyValue);
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

    SVGTests::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGMaskElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isAnimatedLengthAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        setPresentationalHintStyleIsDirty();
        return;
    }

    if (PropertyRegistry::isKnownAttribute(attrName)) {
        if (document().settings().layerBasedSVGEngineEnabled()) {
            if (CheckedPtr maskRenderer = dynamicDowncast<RenderSVGResourceMasker>(renderer())) {
                maskRenderer->invalidateMask();
                maskRenderer->repaintClientsOfReferencedSVGResources();
                return;
            }
        }

        updateSVGRendererForElementChange();
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

void SVGMaskElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);

    if (change.source == ChildChange::Source::Parser)
        return;

    if (document().settings().layerBasedSVGEngineEnabled()) {
        if (CheckedPtr maskRenderer = dynamicDowncast<RenderSVGResourceMasker>(renderer())) {
            maskRenderer->invalidateMask();
            maskRenderer->repaintClientsOfReferencedSVGResources();
        }
    }

    updateSVGRendererForElementChange();
}

RenderPtr<RenderElement> SVGMaskElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return createRenderer<RenderSVGResourceMasker>(*this, WTFMove(style));
    return createRenderer<LegacyRenderSVGResourceMasker>(*this, WTFMove(style));
}

FloatRect SVGMaskElement::calculateMaskContentRepaintRect(RepaintRectCalculation repaintRectCalculation)
{
    ASSERT(renderer());
    auto transformationMatrixFromChild = [&](const RenderLayerModelObject& child) -> std::optional<AffineTransform> {
        if (!document().settings().layerBasedSVGEngineEnabled())
            return std::nullopt;

        if (!(renderer()->isTransformed() || child.isTransformed()) || !child.hasLayer())
            return std::nullopt;

        ASSERT(child.isSVGLayerAwareRenderer());
        ASSERT(!child.isRenderSVGRoot());

        auto transform = SVGLayerTransformComputation(child).computeAccumulatedTransform(downcast<RenderLayerModelObject>(renderer()), TransformState::TrackSVGCTMMatrix);
        return transform.isIdentity() ? std::nullopt : std::make_optional(WTFMove(transform));
    };
    FloatRect maskRepaintRect;
    for (auto* childNode = firstChild(); childNode; childNode = childNode->nextSibling()) {
        CheckedPtr renderer = childNode->renderer();
        if (!childNode->isSVGElement() || !renderer)
            continue;
        const auto& style = renderer->style();
        if (style.display() == DisplayType::None || style.usedVisibility() != Visibility::Visible)
            continue;
        auto r = renderer->repaintRectInLocalCoordinates(repaintRectCalculation);
        if (auto transform = transformationMatrixFromChild(downcast<RenderLayerModelObject>(*renderer)))
            r = transform->mapRect(r);
        maskRepaintRect.unite(r);
    }
    return maskRepaintRect;
}

}
