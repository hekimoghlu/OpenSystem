/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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

#include "FormController.h"
#include "FormListedElement.h"
#include "HTMLElement.h"
#include "ValidationMessage.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/TriState.h>

namespace WebCore {

class HTMLMaybeFormAssociatedCustomElement;

class ValidatedFormListedElement : public FormListedElement {
    WTF_MAKE_TZONE_ALLOCATED(ValidatedFormListedElement);
    WTF_MAKE_NONCOPYABLE(ValidatedFormListedElement);
    friend class DelayedUpdateValidityScope;
    friend class HTMLMaybeFormAssociatedCustomElement;
public:
    ValidatedFormListedElement(HTMLFormElement*);
    virtual ~ValidatedFormListedElement();

    // "willValidate" means "is a candidate for constraint validation".
    WEBCORE_EXPORT bool willValidate() const override;
    void updateVisibleValidationMessage(Ref<HTMLElement> validationAnchor);
    void hideVisibleValidationMessage();
    WEBCORE_EXPORT bool checkValidity(Vector<RefPtr<ValidatedFormListedElement>>* unhandledInvalidControls = nullptr);
    bool reportValidity();
    RefPtr<HTMLElement> focusableValidationAnchorElement();
    void reportNonFocusableControlError();
    WEBCORE_EXPORT void focusAndShowValidationMessage(Ref<HTMLElement> validationAnchor);
    bool isShowingValidationMessage() const;
    WEBCORE_EXPORT bool isFocusingWithValidationMessage() const;
    // This must be called when a validation constraint or control value is changed.
    void updateValidity();
    WEBCORE_EXPORT void setCustomValidity(const String&) override;

    void setDisabledByAncestorFieldset(bool isDisabled);
    virtual void reset() { }

    virtual bool supportsReadOnly() const { return false; }
    bool isDisabled() const { return m_disabled || m_disabledByAncestorFieldset; }
    bool isReadOnly() const { return supportsReadOnly() && m_hasReadOnlyAttribute; }
    bool isMutable() const { return !isDisabled() && !isReadOnly(); }

    // This must be called any time the result of willValidate() has changed.
    bool isValidFormControlElement() const { return m_isValid; }

    bool isEnumeratable() const override { return false; }

    bool wasInteractedWithSinceLastFormSubmitEvent() const { return m_wasInteractedWithSinceLastFormSubmitEvent; }
    void setInteractedWithSinceLastFormSubmitEvent(bool);

    bool matchesValidPseudoClass() const;
    bool matchesInvalidPseudoClass() const;
    bool matchesUserInvalidPseudoClass() const;
    bool matchesUserValidPseudoClass() const;

    bool isCandidateForSavingAndRestoringState() const;
    virtual bool shouldAutocomplete() const;
    virtual bool shouldSaveAndRestoreFormControlState() const { return false; }
    virtual FormControlState saveFormControlState() const;
    virtual void restoreFormControlState(const FormControlState&) { } // Called only if state is not empty.
    virtual const AtomString& formControlType() const = 0;

protected:
    bool hasDisabledAttribute() const { return m_disabled; }
    virtual bool computeWillValidate() const;
    virtual bool readOnlyBarsFromConstraintValidation() const { return false; }
    void updateWillValidateAndValidity();
    bool disabledByAncestorFieldset() const { return m_disabledByAncestorFieldset; }

    bool validationMessageShadowTreeContains(const Node&) const;

    void insertedIntoAncestor(Node::InsertionType, ContainerNode&);
    void didFinishInsertingNode();
    void removedFromAncestor(Node::RemovalType, ContainerNode&);
    void parseAttribute(const QualifiedName&, const AtomString&);
    void parseDisabledAttribute(const AtomString&);
    void parseReadOnlyAttribute(const AtomString&);

    virtual void disabledStateChanged();
    virtual void readOnlyStateChanged();

    void willChangeForm() override;
    void didChangeForm() override;
    void formWillBeDestroyed() final;
    bool belongsToFormThatIsBeingDestroyed() const { return m_belongsToFormThatIsBeingDestroyed; }

    void setDataListAncestorState(TriState);
    void syncWithFieldsetAncestors(ContainerNode* insertionNode);
    void restoreFormControlStateIfNecessary();

private:
    bool computeIsDisabledByFieldsetAncestor() const;
    void setDisabledInternal(bool disabled, bool disabledByAncestorFieldset);
    virtual HTMLElement* validationAnchorElement() = 0;

    void startDelayingUpdateValidity() { ++m_delayedUpdateValidityCount; }
    void endDelayingUpdateValidity();

    RefPtr<ValidationMessage> m_validationMessage;

    // Cache of validity()->valid().
    // But "candidate for constraint validation" doesn't affect isValid.
    bool m_isValid : 1 { true };

    // The initial value of willValidate depends on the derived class.
    // We can't initialize it with a virtual function in the constructor.
    // willValidate is not deterministic as long as willValidateInitialized is false.
    mutable bool m_willValidate : 1 { true };
    mutable bool m_willValidateInitialized : 1 { false };

    bool m_disabled : 1 { false };
    bool m_disabledByAncestorFieldset : 1 { false };

    bool m_hasReadOnlyAttribute : 1 { false };
    bool m_wasInteractedWithSinceLastFormSubmitEvent : 1 { false };
    bool m_belongsToFormThatIsBeingDestroyed : 1 { false };
    bool m_isFocusingWithValidationMessage { false };

    mutable TriState m_isInsideDataList : 2 { TriState::Indeterminate };

    unsigned m_delayedUpdateValidityCount { 0 };
};

class DelayedUpdateValidityScope {
public:
    explicit DelayedUpdateValidityScope(ValidatedFormListedElement& element)
        : m_element { element }
    {
        m_element->startDelayingUpdateValidity();
    }
    
    ~DelayedUpdateValidityScope()
    {
        m_element->endDelayingUpdateValidity();
    }

private:
    Ref<ValidatedFormListedElement> m_element;
};

} // namespace WebCore
