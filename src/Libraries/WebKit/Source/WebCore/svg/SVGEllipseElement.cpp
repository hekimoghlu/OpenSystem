/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#include "SVGEllipseElement.h"

#include "LegacyRenderSVGEllipse.h"
#include "LegacyRenderSVGResource.h"
#include "NodeName.h"
#include "RenderSVGEllipse.h"
#include "SVGElementInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGEllipseElement);

inline SVGEllipseElement::SVGEllipseElement(const QualifiedName& tagName, Document& document)
    : SVGGeometryElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::ellipseTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::cxAttr, &SVGEllipseElement::m_cx>();
        PropertyRegistry::registerProperty<SVGNames::cyAttr, &SVGEllipseElement::m_cy>();
        PropertyRegistry::registerProperty<SVGNames::rxAttr, &SVGEllipseElement::m_rx>();
        PropertyRegistry::registerProperty<SVGNames::ryAttr, &SVGEllipseElement::m_ry>();
    });
}    

Ref<SVGEllipseElement> SVGEllipseElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGEllipseElement(tagName, document));
}

void SVGEllipseElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGParsingError parseError = NoError;

    switch (name.nodeName()) {
    case AttributeNames::cxAttr:
        Ref { m_cx }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError));
        break;
    case AttributeNames::cyAttr:
        Ref { m_cy }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError));
        break;
    case AttributeNames::rxAttr:
        Ref { m_rx }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError, SVGLengthNegativeValuesMode::Forbid));
        break;
    case AttributeNames::ryAttr:
        Ref { m_ry }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError, SVGLengthNegativeValuesMode::Forbid));
        break;
    default:
        break;
    }
    reportAttributeParsingError(parseError, name, newValue);
    SVGGeometryElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGEllipseElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        setPresentationalHintStyleIsDirty();
        invalidateResourceImageBuffersIfNeeded();
        return;
    }

    SVGGeometryElement::svgAttributeChanged(attrName);
}

RenderPtr<RenderElement> SVGEllipseElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return createRenderer<RenderSVGEllipse>(*this, WTFMove(style));
    return createRenderer<LegacyRenderSVGEllipse>(*this, WTFMove(style));
}

}
