/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#include "SVGRectElement.h"

#include "LegacyRenderSVGRect.h"
#include "LegacyRenderSVGResource.h"
#include "NodeName.h"
#include "RenderSVGRect.h"
#include "SVGElementInlines.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGRectElement);

inline SVGRectElement::SVGRectElement(const QualifiedName& tagName, Document& document)
    : SVGGeometryElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::rectTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::xAttr, &SVGRectElement::m_x>();
        PropertyRegistry::registerProperty<SVGNames::yAttr, &SVGRectElement::m_y>();
        PropertyRegistry::registerProperty<SVGNames::widthAttr, &SVGRectElement::m_width>();
        PropertyRegistry::registerProperty<SVGNames::heightAttr, &SVGRectElement::m_height>();
        PropertyRegistry::registerProperty<SVGNames::rxAttr, &SVGRectElement::m_rx>();
        PropertyRegistry::registerProperty<SVGNames::ryAttr, &SVGRectElement::m_ry>();
    });
}

Ref<SVGRectElement> SVGRectElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGRectElement(tagName, document));
}

SVGAnimatedProperty* SVGRectElement::propertyForAttribute(const QualifiedName& name) const
{
    if (name == SVGNames::xAttr)
        return m_x.ptr();
    if (name == SVGNames::yAttr)
        return m_y.ptr();
    if (name == SVGNames::widthAttr)
        return m_width.ptr();
    if (name == SVGNames::heightAttr)
        return m_height.ptr();
    if (name == SVGNames::rxAttr)
        return m_rx.ptr();
    if (name == SVGNames::ryAttr)
        return m_ry.ptr();
    return nullptr;
}

void SVGRectElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGParsingError parseError = NoError;

    switch (name.nodeName()) {
    case AttributeNames::xAttr:
        Ref { m_x }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError));
        break;
    case AttributeNames::yAttr:
        Ref { m_y }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError));
        break;
    case AttributeNames::rxAttr:
        Ref { m_rx }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError, SVGLengthNegativeValuesMode::Forbid));
        break;
    case AttributeNames::ryAttr:
        Ref { m_ry }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError, SVGLengthNegativeValuesMode::Forbid));
        break;
    case AttributeNames::widthAttr:
        Ref { m_width }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError, SVGLengthNegativeValuesMode::Forbid));
        break;
    case AttributeNames::heightAttr:
        Ref { m_height }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError, SVGLengthNegativeValuesMode::Forbid));
        break;
    default:
        break;
    }
    reportAttributeParsingError(parseError, name, newValue);
    SVGGeometryElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGRectElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        setPresentationalHintStyleIsDirty();
        invalidateResourceImageBuffersIfNeeded();
        return;
    }

    SVGGeometryElement::svgAttributeChanged(attrName);
}

RenderPtr<RenderElement> SVGRectElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return createRenderer<RenderSVGRect>(*this, WTFMove(style));
    return createRenderer<LegacyRenderSVGRect>(*this, WTFMove(style));
}

}
