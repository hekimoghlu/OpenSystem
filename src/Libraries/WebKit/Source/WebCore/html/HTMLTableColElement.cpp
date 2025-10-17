/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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
#include "HTMLTableColElement.h"

#include "CSSPropertyNames.h"
#include "ElementInlines.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "HTMLTableElement.h"
#include "RenderTableCol.h"
#include "Text.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLTableColElement);

const unsigned defaultSpan { 1 };
const unsigned minSpan { 1 };
const unsigned maxSpan { 1000 };

using namespace HTMLNames;

inline HTMLTableColElement::HTMLTableColElement(const QualifiedName& tagName, Document& document)
    : HTMLTablePartElement(tagName, document)
    , m_span(1)
{
}

Ref<HTMLTableColElement> HTMLTableColElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLTableColElement(tagName, document));
}

bool HTMLTableColElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    if (name == widthAttr)
        return true;
    return HTMLTablePartElement::hasPresentationalHintsForAttribute(name);
}

void HTMLTableColElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    if (name == widthAttr)
        addHTMLMultiLengthToStyle(style, CSSPropertyWidth, value);
    else if (name == heightAttr)
        addHTMLMultiLengthToStyle(style, CSSPropertyHeight, value);
    else
        HTMLTablePartElement::collectPresentationalHintsForAttribute(name, value, style);
}

void HTMLTableColElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    HTMLTablePartElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);

    if (name == spanAttr) {
        m_span = clampHTMLNonNegativeIntegerToRange(newValue, minSpan, maxSpan, defaultSpan);
        if (CheckedPtr col = dynamicDowncast<RenderTableCol>(renderer()))
            col->updateFromElement();
    } else if (name == widthAttr) {
        if (!newValue.isEmpty()) {
            if (CheckedPtr col = dynamicDowncast<RenderTableCol>(renderer())) {
                int newWidth = parseHTMLInteger(newValue).value_or(0);
                if (newWidth != col->width())
                    col->setNeedsLayoutAndPrefWidthsRecalc();
            }
        }
    }
}

const MutableStyleProperties* HTMLTableColElement::additionalPresentationalHintStyle() const
{
    if (!hasTagName(colgroupTag))
        return nullptr;
    if (auto table = findParentTable())
        return table->additionalGroupStyle(false);
    return nullptr;
}

void HTMLTableColElement::setSpan(unsigned span)
{
    setUnsignedIntegralAttribute(spanAttr, limitToOnlyHTMLNonNegative(span, defaultSpan));
}

String HTMLTableColElement::width() const
{
    return attributeWithoutSynchronization(widthAttr);
}

}
