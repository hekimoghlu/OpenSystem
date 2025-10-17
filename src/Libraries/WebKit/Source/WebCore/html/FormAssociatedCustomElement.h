/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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

#include "CustomElementFormValue.h"
#include "HTMLMaybeFormAssociatedCustomElement.h"
#include "ValidatedFormListedElement.h"
#include "ValidityStateFlags.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FormAssociatedCustomElement final : public ValidatedFormListedElement {
    WTF_MAKE_TZONE_ALLOCATED(FormAssociatedCustomElement);
    WTF_MAKE_NONCOPYABLE(FormAssociatedCustomElement);
public:
    static Ref<FormAssociatedCustomElement> create(HTMLMaybeFormAssociatedCustomElement&);

    FormAssociatedCustomElement(HTMLMaybeFormAssociatedCustomElement&);
    virtual ~FormAssociatedCustomElement();

    bool isFormListedElement() const final { return true; }
    bool isValidatedFormListedElement() const final { return true; }

    FormAssociatedElement* asFormAssociatedElement() final { return this; }
    FormListedElement* asFormListedElement() final { return this; }
    ValidatedFormListedElement* asValidatedFormListedElement() final { return this; }

    HTMLElement& asHTMLElement() final { return *m_element.get(); }
    const HTMLElement& asHTMLElement() const final { return *m_element.get(); }

    void reset() final;
    bool isEnumeratable() const final;

    void setFormValue(CustomElementFormValue&& submissionValue, std::optional<CustomElementFormValue>&& state);
    ExceptionOr<void> setValidity(ValidityStateFlags, String&& message, HTMLElement* validationAnchor);
    String validationMessage() const final;

    void finishParsingChildren();

    bool computeValidity() const final;
    bool appendFormData(DOMFormData&) final;

    void willUpgrade();
    void didUpgrade();

    const AtomString& formControlType() const final;
    bool shouldSaveAndRestoreFormControlState() const final;
    FormControlState saveFormControlState() const final;
    void restoreFormControlState(const FormControlState&) final;

protected:
    void disabledStateChanged() final;
    bool readOnlyBarsFromConstraintValidation() const final { return true; }

private:
    void refFormAssociatedElement() const final { m_element->ref(); }
    void derefFormAssociatedElement() const final { m_element->deref(); }

    HTMLElement* validationAnchorElement() final;
    void didChangeForm() final;
    void invalidateElementsCollectionCachesInAncestors();

    bool hasBadInput() const final { return m_validityStateFlags.badInput; }
    bool patternMismatch() const final { return m_validityStateFlags.patternMismatch; }
    bool rangeOverflow() const final { return m_validityStateFlags.rangeOverflow; }
    bool rangeUnderflow() const final { return m_validityStateFlags.rangeUnderflow; }
    bool stepMismatch() const final { return m_validityStateFlags.stepMismatch; }
    bool tooShort() const final { return m_validityStateFlags.tooShort; }
    bool tooLong() const final { return m_validityStateFlags.tooLong; }
    bool typeMismatch() const final { return m_validityStateFlags.typeMismatch; }
    bool valueMissing() const final { return m_validityStateFlags.valueMissing; }

    WeakPtr<HTMLMaybeFormAssociatedCustomElement, WeakPtrImplWithEventTargetData> m_element;
    ValidityStateFlags m_validityStateFlags;
    WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData> m_validationAnchor { nullptr };
    CustomElementFormValue m_submissionValue { nullptr };
    CustomElementFormValue m_state { nullptr };
};

} // namespace WebCore
