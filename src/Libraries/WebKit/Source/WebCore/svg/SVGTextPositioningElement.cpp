/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
#include "SVGTextPositioningElement.h"

#include "LegacyRenderSVGResource.h"
#include "NodeName.h"
#include "RenderSVGInline.h"
#include "RenderSVGText.h"
#include "SVGAltGlyphElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGNames.h"
#include "SVGTRefElement.h"
#include "SVGTSpanElement.h"
#include "SVGTextElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGTextPositioningElement);

SVGTextPositioningElement::SVGTextPositioningElement(const QualifiedName& tagName, Document& document, UniqueRef<SVGPropertyRegistry>&& propertyRegistry)
    : SVGTextContentElement(tagName, document, WTFMove(propertyRegistry))
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::xAttr, &SVGTextPositioningElement::m_x>();
        PropertyRegistry::registerProperty<SVGNames::yAttr, &SVGTextPositioningElement::m_y>();
        PropertyRegistry::registerProperty<SVGNames::dxAttr, &SVGTextPositioningElement::m_dx>();
        PropertyRegistry::registerProperty<SVGNames::dyAttr, &SVGTextPositioningElement::m_dy>();
        PropertyRegistry::registerProperty<SVGNames::rotateAttr, &SVGTextPositioningElement::m_rotate>();
    });
}

void SVGTextPositioningElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::xAttr:
        Ref { m_x }->baseVal()->parse(newValue);
        break;
    case AttributeNames::yAttr:
        Ref { m_y }->baseVal()->parse(newValue);
        break;
    case AttributeNames::dxAttr:
        Ref { m_dx }->baseVal()->parse(newValue);
        break;
    case AttributeNames::dyAttr:
        Ref { m_dy }->baseVal()->parse(newValue);
        break;
    case AttributeNames::rotateAttr:
        Ref { m_rotate }->baseVal()->parse(newValue);
        break;
    default:
        break;
    }

    SVGTextContentElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGTextPositioningElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    if (name == SVGNames::xAttr || name == SVGNames::yAttr)
        return;
    SVGTextContentElement::collectPresentationalHintsForAttribute(name, value, style);
}

bool SVGTextPositioningElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    if (name == SVGNames::xAttr || name == SVGNames::yAttr)
        return false;
    return SVGTextContentElement::hasPresentationalHintsForAttribute(name);
}

void SVGTextPositioningElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);

        if (attrName != SVGNames::rotateAttr)
            updateRelativeLengthsInformation();

        if (CheckedPtr renderer = this->renderer()) {
            if (CheckedPtr textAncestor = RenderSVGText::locateRenderSVGTextAncestor(*renderer))
                textAncestor->setNeedsPositioningValuesUpdate();
        }
        updateSVGRendererForElementChange();
        invalidateResourceImageBuffersIfNeeded();
        return;
    }

    SVGTextContentElement::svgAttributeChanged(attrName);
}

SVGTextPositioningElement* SVGTextPositioningElement::elementFromRenderer(RenderBoxModelObject& renderer)
{
    if (!is<RenderSVGText>(renderer) && !is<RenderSVGInline>(renderer))
        return nullptr;

    ASSERT(renderer.element());
    SVGElement& element = downcast<SVGElement>(*renderer.element());

    if (!is<SVGTextElement>(element)
        && !is<SVGTSpanElement>(element)
        && !is<SVGAltGlyphElement>(element)
        && !is<SVGTRefElement>(element))
        return nullptr;

    // FIXME: This should use downcast<>().
    return &static_cast<SVGTextPositioningElement&>(element);
}

}
