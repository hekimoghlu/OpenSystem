/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#include "HTMLTablePartElement.h"

#include "CSSImageValue.h"
#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "Document.h"
#include "ElementAncestorIteratorInlines.h"
#include "HTMLNames.h"
#include "HTMLTableElement.h"
#include "MutableStyleProperties.h"
#include "NodeName.h"
#include "StyleProperties.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLTablePartElement);

using namespace HTMLNames;

bool HTMLTablePartElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    switch (name.nodeName()) {
    case AttributeNames::bgcolorAttr:
    case AttributeNames::backgroundAttr:
    case AttributeNames::valignAttr:
    case AttributeNames::heightAttr:
        return true;
    default:
        break;
    }
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLTablePartElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    switch (name.nodeName()) {
    case AttributeNames::bgcolorAttr:
        addHTMLColorToStyle(style, CSSPropertyBackgroundColor, value);
        break;
    case AttributeNames::backgroundAttr:
        if (!StringView(value).containsOnly<isASCIIWhitespace<UChar>>())
            style.setProperty(CSSProperty(CSSPropertyBackgroundImage, CSSImageValue::create(document().completeURL(value), LoadedFromOpaqueSource::No)));
        break;
    case AttributeNames::valignAttr:
        if (equalLettersIgnoringASCIICase(value, "top"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyVerticalAlign, CSSValueTop);
        else if (equalLettersIgnoringASCIICase(value, "middle"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyVerticalAlign, CSSValueMiddle);
        else if (equalLettersIgnoringASCIICase(value, "bottom"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyVerticalAlign, CSSValueBottom);
        else if (equalLettersIgnoringASCIICase(value, "baseline"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyVerticalAlign, CSSValueBaseline);
        else
            addPropertyToPresentationalHintStyle(style, CSSPropertyVerticalAlign, value);
        break;
    case AttributeNames::alignAttr:
        if (equalLettersIgnoringASCIICase(value, "middle"_s) || equalLettersIgnoringASCIICase(value, "center"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyTextAlign, CSSValueWebkitCenter);
        else if (equalLettersIgnoringASCIICase(value, "absmiddle"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyTextAlign, CSSValueCenter);
        else if (equalLettersIgnoringASCIICase(value, "left"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyTextAlign, CSSValueWebkitLeft);
        else if (equalLettersIgnoringASCIICase(value, "right"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyTextAlign, CSSValueWebkitRight);
        else
            addPropertyToPresentationalHintStyle(style, CSSPropertyTextAlign, value);
        break;
    case AttributeNames::heightAttr:
        if (!value.isEmpty())
            addHTMLLengthToStyle(style, CSSPropertyHeight, value);
        break;
    default:
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
        break;
    }
}

RefPtr<const HTMLTableElement> HTMLTablePartElement::findParentTable() const
{
    return ancestorsOfType<HTMLTableElement>(*this).first();
}

}
