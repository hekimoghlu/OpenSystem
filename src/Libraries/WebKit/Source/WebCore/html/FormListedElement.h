/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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

#include "FormAssociatedElement.h"
#include "Node.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ContainerNode;
class DOMFormData;
class Document;
class FormAttributeTargetObserver;
class HTMLElement;
class HTMLFormElement;
class ValidityState;

// https://html.spec.whatwg.org/multipage/forms.html#category-listed
class FormListedElement : public FormAssociatedElement {
    WTF_MAKE_TZONE_ALLOCATED(FormListedElement);
    WTF_MAKE_NONCOPYABLE(FormListedElement);
public:
    virtual ~FormListedElement();

    ValidityState* validity();

    virtual bool isValidatedFormListedElement() const = 0;
    virtual bool isEnumeratable() const = 0;

    // Returns the 'name' attribute value. If this element has no name
    // attribute, it returns an empty string instead of null string.
    // Note that the 'name' IDL attribute doesn't use this function.
    virtual const AtomString& name() const;

    // Override in derived classes to get the encoded name=value pair for submitting.
    // Return true for a successful control (see HTML4-17.13.2).
    virtual bool appendFormData(DOMFormData&) { return false; }

    void formWillBeDestroyed() override;

    void resetFormOwner() final;

    void formOwnerRemovedFromTree(const Node&);

    // ValidityState attribute implementations
    bool badInput() const { return hasBadInput(); }
    bool customError() const;

    // Implementations of patternMismatch, rangeOverflow, rangerUnderflow, stepMismatch, tooShort, tooLong and valueMissing must call willValidate.
    virtual bool hasBadInput() const;
    virtual bool patternMismatch() const;
    virtual bool rangeOverflow() const;
    virtual bool rangeUnderflow() const;
    virtual bool stepMismatch() const;
    virtual bool tooShort() const;
    virtual bool tooLong() const;
    virtual bool typeMismatch() const;
    virtual bool valueMissing() const;
    virtual String validationMessage() const;
    virtual bool computeValidity() const;
    virtual void setCustomValidity(const String&);

    void formAttributeTargetChanged();

    virtual FormListedElement* asFormListedElement() = 0;
    virtual ValidatedFormListedElement* asValidatedFormListedElement() = 0;

protected:
    FormListedElement(HTMLFormElement*);

    void clearForm() { setForm(nullptr); }

    void didMoveToNewDocument();
    void elementInsertedIntoAncestor(Element&, Node::InsertionType) override;
    void elementRemovedFromAncestor(Element&, Node::RemovalType) override;
    void parseAttribute(const QualifiedName&, const AtomString&);
    void parseFormAttribute(const AtomString&);

    // If you add an override of willChangeForm() or didChangeForm() to a class
    // derived from this one, you will need to add a call to setForm(0) to the
    // destructor of that class.
    virtual void willChangeForm();
    virtual void didChangeForm();

    String customValidationMessage() const;

private:
    void setFormInternal(RefPtr<HTMLFormElement>&&) final;
    // "willValidate" means "is a candidate for constraint validation".
    virtual bool willValidate() const = 0;

    void resetFormAttributeTargetObserver();

    std::unique_ptr<FormAttributeTargetObserver> m_formAttributeTargetObserver;
    String m_customValidationMessage;
};

} // namespace
