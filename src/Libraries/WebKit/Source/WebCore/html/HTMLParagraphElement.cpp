/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
#include "HTMLParagraphElement.h"

#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "Document.h"
#include "HTMLNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLParagraphElement);

using namespace HTMLNames;

inline HTMLParagraphElement::HTMLParagraphElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(pTag));
}

Ref<HTMLParagraphElement> HTMLParagraphElement::create(Document& document)
{
    return adoptRef(*new HTMLParagraphElement(pTag, document));
}

Ref<HTMLParagraphElement> HTMLParagraphElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLParagraphElement(tagName, document));
}

void HTMLParagraphElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
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
