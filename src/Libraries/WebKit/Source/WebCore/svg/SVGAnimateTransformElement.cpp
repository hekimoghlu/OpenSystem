/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "SVGAnimateTransformElement.h"

#include "SVGNames.h"
#include "SVGTransformable.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGAnimateTransformElement);

inline SVGAnimateTransformElement::SVGAnimateTransformElement(const QualifiedName& tagName, Document& document)
    : SVGAnimateElementBase(tagName, document)
    , m_type(SVGTransformValue::SVG_TRANSFORM_TRANSLATE)
{
    ASSERT(hasTagName(SVGNames::animateTransformTag));
}

Ref<SVGAnimateTransformElement> SVGAnimateTransformElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGAnimateTransformElement(tagName, document));
}

bool SVGAnimateTransformElement::hasValidAttributeType() const
{
    if (!this->targetElement())
        return false;

    if (attributeType() == AttributeType::CSS)
        return false;

    return SVGAnimateElementBase::hasValidAttributeType();
}

void SVGAnimateTransformElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::typeAttr) {
        m_type = SVGTransformable::parseTransformType(newValue).value_or(SVGTransformValue::SVG_TRANSFORM_UNKNOWN);
        if (m_type == SVGTransformValue::SVG_TRANSFORM_MATRIX)
            m_type = SVGTransformValue::SVG_TRANSFORM_UNKNOWN;
    }

    SVGAnimateElementBase::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

String SVGAnimateTransformElement::animateRangeString(const String& string) const
{
    return makeString(SVGTransformValue::prefixForTransformType(m_type), string, ')');
}

}
