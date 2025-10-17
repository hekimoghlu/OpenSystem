/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
#include "HTMLBRElement.h"

#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "HTMLNames.h"
#include "RenderLineBreak.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLBRElement);

using namespace HTMLNames;

HTMLBRElement::HTMLBRElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(brTag));
}

Ref<HTMLBRElement> HTMLBRElement::create(Document& document)
{
    return adoptRef(*new HTMLBRElement(brTag, document));
}

Ref<HTMLBRElement> HTMLBRElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLBRElement(tagName, document));
}

bool HTMLBRElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    if (name == clearAttr)
        return true;
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLBRElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    if (name == clearAttr) {
        // If the string is empty, then don't add the clear property.
        // <br clear> and <br clear=""> are just treated like <br> by Gecko, Mac IE, etc. -dwh
        if (!value.isEmpty()) {
            if (equalLettersIgnoringASCIICase(value, "all"_s))
                addPropertyToPresentationalHintStyle(style, CSSPropertyClear, CSSValueBoth);
            else
                addPropertyToPresentationalHintStyle(style, CSSPropertyClear, value);
        }
    } else
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
}

RenderPtr<RenderElement> HTMLBRElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (style.hasContent() && RenderElement::isContentDataSupported(*style.contentData()))
        return RenderElement::createFor(*this, WTFMove(style));

    return createRenderer<RenderLineBreak>(*this, WTFMove(style));
}

}
