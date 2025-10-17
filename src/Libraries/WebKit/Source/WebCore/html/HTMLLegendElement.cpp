/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#include "HTMLLegendElement.h"

#include "ElementIterator.h"
#include "HTMLFieldSetElement.h"
#include "HTMLNames.h"
#include "SelectionRestorationMode.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLLegendElement);

inline HTMLLegendElement::HTMLLegendElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(HTMLNames::legendTag));
}

Ref<HTMLLegendElement> HTMLLegendElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLLegendElement(tagName, document));
}

HTMLFormElement* HTMLLegendElement::form() const
{
    // According to the specification, If the legend has a fieldset element as
    // its parent, then the form attribute must return the same value as the
    // form attribute on that fieldset element. Otherwise, it must return null.
    RefPtr fieldset = dynamicDowncast<HTMLFieldSetElement>(parentNode());
    return fieldset ? fieldset->form() : nullptr;
}
    
} // namespace
