/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
#include "HTMLPreElement.h"

#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "HTMLNames.h"
#include "MutableStyleProperties.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLPreElement);

using namespace HTMLNames;

inline HTMLPreElement::HTMLPreElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
}

Ref<HTMLPreElement> HTMLPreElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLPreElement(tagName, document));
}

bool HTMLPreElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    if (name == wrapAttr)
        return true;
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLPreElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    if (name == wrapAttr) {
        style.setProperty(CSSPropertyWhiteSpaceCollapse, CSSValuePreserve);
        style.setProperty(CSSPropertyTextWrapMode, CSSValueWrap);
    } else
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
}

}
