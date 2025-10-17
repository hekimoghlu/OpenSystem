/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#include "FormState.h"
#include "FormSubmission.h"
#include "HTMLElement.h"
#include "RadioButtonGroups.h"
#include <memory>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

class DOMFormData;
class Event;
class FormListedElement;
class HTMLFormControlElement;
class HTMLFormControlsCollection;
class HTMLImageElement;
class ValidatedFormListedElement;

class HTMLFormElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLFormElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLFormElement);
public:
    static Ref<HTMLFormElement> create(Document&);
    static Ref<HTMLFormElement> create(const QualifiedName&, Document&);
    virtual ~HTMLFormElement();

    Ref<HTMLFormControlsCollection> elements();
    WEBCORE_EXPORT Ref<HTMLCollection> elementsForNativeBindings();
    Vector<Ref<Element>> namedElements(const AtomString&);
    bool isSupportedPropertyName(const AtomString&);

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    WEBCORE_EXPORT unsigned length() const;
    HTMLElement* item(unsigned index);
    std::optional<std::variant<RefPtr<RadioNodeList>, RefPtr<Element>>> namedItem(const AtomString&);
    Vector<AtomString> supportedPropertyNames() const;

    String enctype() const { return m_attributes.encodingType(); }
    WEBCORE_EXPORT void setEnctype(const AtomString&);

    bool shouldAutocomplete() const;

    WEBCORE_EXPORT void setAutocomplete(const AtomString&);
    WEBCORE_EXPORT const AtomString& autocomplete() const;

    void registerFormListedElement(FormListedElement&);
    void unregisterFormListedElement(FormListedElement&);

    void addInvalidFormControl(const HTMLElement&);
    void removeInvalidFormControlIfNeeded(const HTMLElement&);

    void registerImgElement(HTMLImageElement&);
    void unregisterImgElement(HTMLImageElement&);

    void submitIfPossible(Event*, HTMLFormControlElement* = nullptr, FormSubmissionTrigger = NotSubmittedByJavaScript);
    WEBCORE_EXPORT void submit();
    void submitFromJavaScript();
    ExceptionOr<void> requestSubmit(HTMLElement* submitter);
    WEBCORE_EXPORT void reset();

    void submitImplicitly(Event&, bool fromImplicitSubmissionTrigger);

    String name() const;

    bool noValidate() const;

    String acceptCharset() const { return m_attributes.acceptCharset(); }
    void setAcceptCharset(const String&);

    WEBCORE_EXPORT String action() const;
    WEBCORE_EXPORT void setAction(const AtomString&);

    WEBCORE_EXPORT String method() const;
    WEBCORE_EXPORT void setMethod(const AtomString&);

    DOMTokenList& relList();

    AtomString target() const final;
    AtomString effectiveTarget(const Event*, HTMLFormControlElement* submitter) const;

    bool wasUserSubmitted() const;

    HTMLFormControlElement* findSubmitter(const Event*) const;

    HTMLFormControlElement* defaultButton() const;
    void resetDefaultButton();

    WEBCORE_EXPORT bool checkValidity();
    bool reportValidity();

    RadioButtonGroups& radioButtonGroups() { return m_radioButtonGroups; }

    WEBCORE_EXPORT const Vector<WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData>>& unsafeListedElements() const;
    WEBCORE_EXPORT Vector<Ref<FormListedElement>> copyListedElementsVector() const;
    Vector<Ref<ValidatedFormListedElement>> copyValidatedListedElementsVector() const;
    const Vector<WeakPtr<HTMLImageElement, WeakPtrImplWithEventTargetData>>& imageElements() const { return m_imageElements; }

    StringPairVector textFieldValues() const;

    static HTMLFormElement* findClosestFormAncestor(const Element&);
    
    RefPtr<DOMFormData> constructEntryList(RefPtr<HTMLFormControlElement>&&, Ref<DOMFormData>&&, StringPairVector*);
    
private:
    HTMLFormElement(const QualifiedName&, Document&);

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;
    void finishParsingChildren() final;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool isURLAttribute(const Attribute&) const final;

    void resumeFromDocumentSuspension() final;

    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) final;

    void submit(Event*, bool processingUserGesture, FormSubmissionTrigger, HTMLFormControlElement* submitter = nullptr);

    void submitDialog(Ref<FormSubmission>&&);

    unsigned formElementIndexWithFormAttribute(Element*, unsigned rangeStart, unsigned rangeEnd);
    unsigned formElementIndex(FormListedElement&);

    bool validateInteractively();

    // Validates each of the controls, and stores controls of which 'invalid'
    // event was not canceled to the specified vector. Returns true if there
    // are any invalid controls in this form.
    bool checkInvalidControlsAndCollectUnhandled(Vector<RefPtr<ValidatedFormListedElement>>&);

    RefPtr<HTMLElement> elementFromPastNamesMap(const AtomString&) const;
    void addToPastNamesMap(FormAssociatedElement&, const AtomString& pastName);
#if ASSERT_ENABLED
    void assertItemCanBeInPastNamesMap(FormAssociatedElement&) const;
#endif
    void removeFromPastNamesMap(FormAssociatedElement&);

    bool matchesValidPseudoClass() const final;
    bool matchesInvalidPseudoClass() const final;

    void resetListedFormControlElements();

    RefPtr<HTMLFormControlElement> findSubmitButton(HTMLFormControlElement* submitter, bool needButtonActivation);

    FormSubmission::Attributes m_attributes;
    UncheckedKeyHashMap<AtomString, WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData>> m_pastNamesMap;

    RadioButtonGroups m_radioButtonGroups;
    mutable WeakPtr<HTMLFormControlElement, WeakPtrImplWithEventTargetData> m_defaultButton;

    Vector<WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData>> m_listedElements;
    Vector<WeakPtr<HTMLImageElement, WeakPtrImplWithEventTargetData>> m_imageElements;
    WeakHashSet<HTMLElement, WeakPtrImplWithEventTargetData> m_invalidFormControls;
    WeakPtr<FormSubmission> m_plannedFormSubmission;
    std::unique_ptr<DOMTokenList> m_relList;

    unsigned m_listedElementsBeforeIndex { 0 };
    unsigned m_listedElementsAfterIndex { 0 };

    bool m_wasUserSubmitted { false };
    bool m_isSubmittingOrPreparingForSubmission { false };
    bool m_shouldSubmit { false };

    bool m_isInResetFunction { false };

    bool m_isConstructingEntryList { false };
};

} // namespace WebCore
