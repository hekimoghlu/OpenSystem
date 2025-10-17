/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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

#include "HTMLFormElement.h"

namespace WebCore {

// https://html.spec.whatwg.org/multipage/forms.html#form-associated-element
class FormAssociatedElement {
public:
    void ref() const { refFormAssociatedElement(); }
    void deref() const { derefFormAssociatedElement(); }

    virtual ~FormAssociatedElement() { RELEASE_ASSERT(!m_form); }
    virtual HTMLElement& asHTMLElement() = 0;
    virtual const HTMLElement& asHTMLElement() const = 0;
    virtual bool isFormListedElement() const = 0;

    virtual void formWillBeDestroyed() { m_form = nullptr; }

    HTMLFormElement* form() const { return m_form.get(); }

    void setForm(RefPtr<HTMLFormElement>&&);
    virtual void elementInsertedIntoAncestor(Element&, Node::InsertionType);
    virtual void elementRemovedFromAncestor(Element&, Node::RemovalType);

    virtual FormAssociatedElement* asFormAssociatedElement() = 0;

protected:
    explicit FormAssociatedElement(HTMLFormElement*);

    virtual void resetFormOwner() = 0;
    virtual void setFormInternal(RefPtr<HTMLFormElement>&&);

private:
    virtual void refFormAssociatedElement() const = 0;
    virtual void derefFormAssociatedElement() const = 0;

    WeakPtr<HTMLFormElement, WeakPtrImplWithEventTargetData> m_form;
    WeakPtr<HTMLFormElement, WeakPtrImplWithEventTargetData> m_formSetByParser;
};

inline void FormAssociatedElement::setForm(RefPtr<HTMLFormElement>&& newForm)
{
    if (m_form.get() != newForm)
        setFormInternal(WTFMove(newForm));
}

} // namespace WebCore
