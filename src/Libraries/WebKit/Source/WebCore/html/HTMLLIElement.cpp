/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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
#include "HTMLLIElement.h"

#include "Attribute.h"
#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "ElementAncestorIteratorInlines.h"
#include "HTMLNames.h"
#include "HTMLOListElement.h"
#include "HTMLParserIdioms.h"
#include "HTMLUListElement.h"
#include "RenderListItem.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLLIElement);

using namespace HTMLNames;

HTMLLIElement::HTMLLIElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document, TypeFlag::HasCustomStyleResolveCallbacks)
{
    ASSERT(hasTagName(liTag));
}

Ref<HTMLLIElement> HTMLLIElement::create(Document& document)
{
    return adoptRef(*new HTMLLIElement(liTag, document));
}

Ref<HTMLLIElement> HTMLLIElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLLIElement(tagName, document));
}

bool HTMLLIElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    if (name == typeAttr || name == valueAttr)
        return true;
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLLIElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    if (name == typeAttr) {
        if (value == "a"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueLowerAlpha);
        else if (value == "A"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueUpperAlpha);
        else if (value == "i"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueLowerRoman);
        else if (value == "I"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueUpperRoman);
        else if (value == "1"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyListStyleType, CSSValueDecimal);
        else {
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
        }
    } else if (name == valueAttr) {
        if (auto parsedValue = parseHTMLInteger(value))
            addPropertyToPresentationalHintStyle(style, CSSPropertyCounterSet, makeString("list-item "_s, *parsedValue));
    } else
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
}

}
