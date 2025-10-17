/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#include "SVGLineElement.h"

#include "LegacyRenderSVGResource.h"
#include "LegacyRenderSVGShape.h"
#include "NodeName.h"
#include "RenderSVGShape.h"
#include "SVGLengthValue.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGLineElement);

inline SVGLineElement::SVGLineElement(const QualifiedName& tagName, Document& document)
    : SVGGeometryElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::lineTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::x1Attr, &SVGLineElement::m_x1>();
        PropertyRegistry::registerProperty<SVGNames::y1Attr, &SVGLineElement::m_y1>();
        PropertyRegistry::registerProperty<SVGNames::x2Attr, &SVGLineElement::m_x2>();
        PropertyRegistry::registerProperty<SVGNames::y2Attr, &SVGLineElement::m_y2>();
    });
}

Ref<SVGLineElement> SVGLineElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGLineElement(tagName, document));
}

void SVGLineElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGParsingError parseError = NoError;

    switch (name.nodeName()) {
    case AttributeNames::x1Attr:
        Ref { m_x1 }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError));
        break;
    case AttributeNames::y1Attr:
        Ref { m_y1 }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError));
        break;
    case AttributeNames::x2Attr:
        Ref { m_x2 }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Width, newValue, parseError));
        break;
    case AttributeNames::y2Attr:
        Ref { m_y2 }->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Height, newValue, parseError));
        break;
    default:
        break;
    }
    reportAttributeParsingError(parseError, name, newValue);

    SVGGeometryElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGLineElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        updateRelativeLengthsInformation();

        if (CheckedPtr shape = dynamicDowncast<RenderSVGShape>(renderer()))
            shape->setNeedsShapeUpdate();

        if (CheckedPtr shape = dynamicDowncast<LegacyRenderSVGShape>(renderer()))
            shape->setNeedsShapeUpdate();

        updateSVGRendererForElementChange();
        invalidateResourceImageBuffersIfNeeded();
        return;
    }

    SVGGeometryElement::svgAttributeChanged(attrName);
}

bool SVGLineElement::selfHasRelativeLengths() const
{
    return x1().isRelative()
        || y1().isRelative()
        || x2().isRelative()
        || y2().isRelative();
}

}
