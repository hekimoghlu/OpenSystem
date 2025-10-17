/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#include "SVGGlyphRefElement.h"

#include "NodeName.h"
#include "SVGElementTypeHelpers.h"
#include "SVGGlyphElement.h"
#include "SVGNames.h"
#include "SVGParserUtilities.h"
#include "XLinkNames.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGGlyphRefElement);

inline SVGGlyphRefElement::SVGGlyphRefElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGURIReference(this)
{
    ASSERT(hasTagName(SVGNames::glyphRefTag));
}

Ref<SVGGlyphRefElement> SVGGlyphRefElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGGlyphRefElement(tagName, document));
}

bool SVGGlyphRefElement::hasValidGlyphElement(String& glyphName) const
{
    // FIXME: We only support xlink:href so far.
    // https://bugs.webkit.org/show_bug.cgi?id=64787
    // No need to support glyphRef referencing another node inside a shadow tree.
    auto target = targetElementFromIRIString(getAttribute(SVGNames::hrefAttr, XLinkNames::hrefAttr), document());
    glyphName = target.identifier;
    return is<SVGGlyphElement>(target.element);
}

static float parseFloat(const AtomString& value)
{
    return parseNumber(value).value_or(0);
}

void SVGGlyphRefElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    // FIXME: Is the error handling in parseFloat correct for these attributes?
    switch (name.nodeName()) {
    case AttributeNames::xAttr:
        m_x = parseFloat(newValue);
        break;
    case AttributeNames::yAttr:
        m_y = parseFloat(newValue);
        break;
    case AttributeNames::dxAttr:
        m_dx = parseFloat(newValue);
        break;
    case AttributeNames::dyAttr:
        m_dy = parseFloat(newValue);
        break;
    default:
        break;
    }

    SVGURIReference::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGGlyphRefElement::setX(float x)
{
    setAttribute(SVGNames::xAttr, AtomString::number(x));
}

void SVGGlyphRefElement::setY(float y)
{
    setAttribute(SVGNames::yAttr, AtomString::number(y));
}

void SVGGlyphRefElement::setDx(float dx)
{
    setAttribute(SVGNames::dxAttr, AtomString::number(dx));
}

void SVGGlyphRefElement::setDy(float dy)
{
    setAttribute(SVGNames::dyAttr, AtomString::number(dy));
}

}
