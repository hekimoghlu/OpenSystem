/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#include "HTMLDivElement.h"

#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "HTMLNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLDivElement);

using namespace HTMLNames;

HTMLDivElement::HTMLDivElement(const QualifiedName& tagName, Document& document, OptionSet<TypeFlag> typeFlags)
    : HTMLElement(tagName, document, typeFlags)
{
    ASSERT(hasTagName(divTag));
}

Ref<HTMLDivElement> HTMLDivElement::create(Document& document)
{
    return adoptRef(*new HTMLDivElement(divTag, document));
}

Ref<HTMLDivElement> HTMLDivElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLDivElement(tagName, document));
}

void HTMLDivElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    if (name == alignAttr) {
        if (equalLettersIgnoringASCIICase(value, "middle"_s) || equalLettersIgnoringASCIICase(value, "center"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyTextAlign, CSSValueWebkitCenter);
        else if (equalLettersIgnoringASCIICase(value, "left"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyTextAlign, CSSValueWebkitLeft);
        else if (equalLettersIgnoringASCIICase(value, "right"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyTextAlign, CSSValueWebkitRight);
        else
            addPropertyToPresentationalHintStyle(style, CSSPropertyTextAlign, value);
    } else
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
}

}
