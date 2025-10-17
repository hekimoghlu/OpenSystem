/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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

#include "AutoFillButtonElement.h"
#include "DataListButtonElement.h"
#include "DataListSuggestionPicker.h"
#include "DataListSuggestionsClient.h"
#include "InputType.h"
#include "SpinButtonElement.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DOMFormData;
class TextControlInnerTextElement;

// The class represents types of which UI contain text fields.
// It supports not only the types for BaseTextInputType but also type=number.
class TextFieldInputType : public InputType, protected SpinButtonOwner, protected AutoFillButtonElement::AutoFillButtonOwner
    , private DataListSuggestionsClient, protected DataListButtonElement::DataListButtonOwner
{
    WTF_MAKE_TZONE_ALLOCATED(TextFieldInputType);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextFieldInputType);
public:
    bool valueMissing(const String&) const final;

protected:
    explicit TextFieldInputType(Type, HTMLInputElement&);
    virtual ~TextFieldInputType();
    ShouldCallBaseEventHandler handleKeydownEvent(KeyboardEvent&) override;
    void handleKeydownEventForSpinButton(KeyboardEvent&);
    void handleClickEvent(MouseEvent&) final;

    HTMLElement* containerElement() const final;
    HTMLElement* innerBlockElement() const final;
    RefPtr<TextControlInnerTextElement> innerTextElement() const final;
    HTMLElement* innerSpinButtonElement() const final;
    HTMLElement* autoFillButtonElement() const final;
    HTMLElement* dataListButtonElement() const final;

    virtual bool needsContainer() const;
    void createShadowSubtree() override;
    void removeShadowSubtree() override;
    void attributeChanged(const QualifiedName&) override;
    void disabledStateChanged() final;
    void readOnlyStateChanged() final;
    bool supportsReadOnly() const final;
    void handleFocusEvent(Node* oldFocusedNode, FocusDirection) final;
    void handleBlurEvent() final;
    void setValue(const String&, bool valueChanged, TextFieldEventBehavior, TextControlSetValueSelection) override;
    void updateInnerTextValue() final;
    String sanitizeValue(const String&) const override;

    virtual String convertFromVisibleValue(const String&) const;
    virtual void didSetValueByUserEdit();

private:
    bool isKeyboardFocusable(KeyboardEvent*) const final;
    bool isMouseFocusable() const final;
    bool isEmptyValue() const final;
    void handleBeforeTextInsertedEvent(BeforeTextInsertedEvent&) final;
    void forwardEvent(Event&) final;
    bool shouldSubmitImplicitly(Event&) final;
    RenderPtr<RenderElement> createInputRenderer(RenderStyle&&) override;
    bool shouldUseInputMethod() const override;
    bool shouldRespectListAttribute() override;
    HTMLElement* placeholderElement() const final;
    void updatePlaceholderText() final;
    bool appendFormData(DOMFormData&) const final;
    void subtreeHasChanged() final;
    void capsLockStateMayHaveChanged() final;
    void updateAutoFillButton() final;
    void elementDidBlur() final;

    // SpinButtonOwner functions.
    void focusAndSelectSpinButtonOwner() final;
    bool shouldSpinButtonRespondToMouseEvents() const final;
    void spinButtonStepDown() final;
    void spinButtonStepUp() final;

    // AutoFillButtonElement::AutoFillButtonOwner
    void autoFillButtonElementWasClicked() final;

    bool shouldHaveSpinButton() const;
    bool shouldHaveCapsLockIndicator() const;
    bool shouldDrawCapsLockIndicator() const;
    bool shouldDrawAutoFillButton() const;

    enum class PreserveSelectionRange : bool { No, Yes };
    void createContainer(PreserveSelectionRange = PreserveSelectionRange::Yes);
    void createAutoFillButton(AutoFillButtonType);

    void createDataListDropdownIndicator();
    bool isPresentingAttachedView() const final;
    bool isFocusingWithDataListDropdown() const final;
    void dataListMayHaveChanged() final;
    void displaySuggestions(DataListSuggestionActivationType);
    void closeSuggestions();

    void showPicker() override;

    // DataListSuggestionsClient
    IntRect elementRectInRootViewCoordinates() const final;
    Vector<DataListSuggestion> suggestions() final;
    void didSelectDataListOption(const String&) final;
    void didCloseSuggestions() final;

    void dataListButtonElementWasClicked() final;
    bool m_isFocusingWithDataListDropdown { false };
    RefPtr<DataListButtonElement> m_dataListDropdownIndicator;

    std::pair<String, Vector<DataListSuggestion>> m_cachedSuggestions;
    RefPtr<DataListSuggestionPicker> m_suggestionPicker;

    RefPtr<HTMLElement> m_container;
    RefPtr<HTMLElement> m_innerBlock;
    RefPtr<TextControlInnerTextElement> m_innerText;
    RefPtr<HTMLElement> m_placeholder;
    RefPtr<SpinButtonElement> m_innerSpinButton;
    RefPtr<HTMLElement> m_capsLockIndicator;
    RefPtr<HTMLElement> m_autoFillButton;
};

} // namespace WebCore
