/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#include "HTMLOListElement.h"

#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "NodeName.h"
#include "RenderListItem.h"
#include <wtf/TZoneMallocInlines.h>

// FIXME: There should be a standard way to turn a std::expected into a Optional.
// Maybe we should put this into the header file for Expected and give it a better name.
template<typename T, typename E> inline std::optional<T> optionalValue(Expected<T, E>&& expected)
{
    return expected ? std::optional<T>(WTFMove(expected.value())) : std::nullopt;
}

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLOListElement);

using namespace HTMLNames;

inline HTMLOListElement::HTMLOListElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(olTag));
}

Ref<HTMLOListElement> HTMLOListElement::create(Document& document)
{
    return adoptRef(*new HTMLOListElement(olTag, document));
}

Ref<HTMLOListElement> HTMLOListElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLOListElement(tagName, document));
}

bool HTMLOListElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    if (name == typeAttr)
        return true;
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLOListElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
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
    } else
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
}

void HTMLOListElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::startAttr: {
        int oldStart = start();
        m_start = optionalValue(parseHTMLInteger(newValue));
        if (oldStart == start())
            return;
        RenderListItem::updateItemValuesForOrderedList(*this);
        break;
    }
    case AttributeNames::reversedAttr: {
        bool reversed = !newValue.isNull();
        if (reversed == m_isReversed)
            return;
        m_isReversed = reversed;
        RenderListItem::updateItemValuesForOrderedList(*this);
        break;
    }
    default:
        HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
        break;
    }
}

void HTMLOListElement::setStartForBindings(int start)
{
    setIntegralAttribute(startAttr, start);
}

unsigned HTMLOListElement::itemCount() const
{
    if (!m_itemCount)
        m_itemCount = RenderListItem::itemCountForOrderedList(*this);
    return m_itemCount.value();
}

}
