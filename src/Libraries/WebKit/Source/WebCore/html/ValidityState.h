/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
#pragma once

#include "FormListedElement.h"
#include "HTMLElement.h"

namespace WebCore {

// ValidityState is not a separate object, but rather an interface of FormListedElement that
// is published as part of the DOM. We could implement this as a base class of FormListedElement,
// but that would have a small runtime cost, and no significant benefit. We'd prefer to implement this
// as a typedef of FormListedElement, but that would require changes to bindings generation.
class ValidityState : public FormListedElement {
public:
    Element* element() { return &asHTMLElement(); }
    Node* opaqueRootConcurrently() { return &asHTMLElement(); }
};

inline ValidityState* FormListedElement::validity()
{
    // Because ValidityState adds nothing to FormListedElement, we rely on it being safe
    // to cast a FormListedElement like this, even though it's not actually a ValidityState.
    return static_cast<ValidityState*>(this);
}

} // namespace WebCore
