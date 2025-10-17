/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "HTMLUListElement.h"

#include "CSSPropertyNames.h"
#include "HTMLNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLUListElement);

using namespace HTMLNames;

HTMLUListElement::HTMLUListElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(ulTag));
}

Ref<HTMLUListElement> HTMLUListElement::create(Document& document)
{
    return adoptRef(*new HTMLUListElement(ulTag, document));
}

Ref<HTMLUListElement> HTMLUListElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLUListElement(tagName, document));
}

bool HTMLUListElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    if (name == typeAttr)
        return true;
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLUListElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    if (name == typeAttr) {
        auto valueLowerCase = value.convertToASCIILowercase();
        if (valueLowerCase == "disc"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueDisc);
        else if (valueLowerCase == "circle"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueCircle);
        else if (valueLowerCase == "round"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueRound);
        else if (valueLowerCase == "square"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueSquare);
        else if (valueLowerCase == "none"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueNone);
    } else
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
}

}
