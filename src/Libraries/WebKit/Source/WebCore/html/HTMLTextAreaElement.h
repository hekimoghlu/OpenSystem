/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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

#include "HTMLTextFormControlElement.h"

namespace WebCore {

class BeforeTextInsertedEvent;
class RenderTextControlMultiLine;

enum class SelectionRestorationMode : uint8_t;

class HTMLTextAreaElement final : public HTMLTextFormControlElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLTextAreaElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLTextAreaElement);
public:
    WEBCORE_EXPORT static Ref<HTMLTextAreaElement> create(Document&);
    static Ref<HTMLTextAreaElement> create(const QualifiedName&, Document&, HTMLFormElement*);

    unsigned rows() const { return m_rows; }
    WEBCORE_EXPORT void setRows(unsigned);
    unsigned cols() const { return m_cols; }
    WEBCORE_EXPORT void setCols(unsigned);
    WEBCORE_EXPORT String defaultValue() const;
    WEBCORE_EXPORT void setDefaultValue(String&&);
    WEBCORE_EXPORT String value() const final;
    WEBCORE_EXPORT ExceptionOr<void> setValue(const String&, TextFieldEventBehavior = DispatchNoEvent, TextControlSetValueSelection = TextControlSetValueSelection::SetSelectionToEnd) final;
    unsigned textLength() const { return value().length(); }
    String validationMessage() const final;

    void setSelectionRangeForBindings(unsigned start, unsigned end, const String& direction);

    WEBCORE_EXPORT RefPtr<TextControlInnerTextElement> innerTextElement() const final;

    bool shouldSaveAndRestoreFormControlState() const final { return true; }

    bool isDevolvableWidget() const override { return true; }

    bool dirAutoUsesValue() const final { return true; }

private:
    HTMLTextAreaElement(Document&, HTMLFormElement*);

    void didAddUserAgentShadowRoot(ShadowRoot&) final;

    static String sanitizeUserInputValue(const String& proposedValue, unsigned maxLength);
    void handleBeforeTextInsertedEvent(BeforeTextInsertedEvent&) const;
    void updateValue() const;
    void setNonDirtyValue(const String&, TextControlSetValueSelection);
    void setValueCommon(const String&, TextFieldEventBehavior, TextControlSetValueSelection);

    bool supportsReadOnly() const final { return true; }

    bool supportsPlaceholder() const final { return true; }
    HTMLElement* placeholderElement() const final { return m_placeholder.get(); }
    RefPtr<HTMLElement> protectedPlaceholderElement() const;
    void updatePlaceholderText() final;
    bool isEmptyValue() const final { return value().isEmpty(); }

    bool isOptionalFormControl() const final { return !isRequiredFormControl(); }
    bool isRequiredFormControl() const final { return isRequired(); }

    void defaultEventHandler(Event&) final;
    
    void subtreeHasChanged() final;

    bool isEnumeratable() const final { return true; }
    bool isLabelable() const final { return true; }

    bool isInteractiveContent() const final { return true; }

    const AtomString& formControlType() const final;

    FormControlState saveFormControlState() const final;
    void restoreFormControlState(const FormControlState&) final;

    bool isTextField() const final { return true; }

    void childrenChanged(const ChildChange&) final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    bool appendFormData(DOMFormData&) final;
    void reset() final;
    bool hasCustomFocusLogic() const final { return true; }
    int defaultTabIndex() const final { return 0; }
    bool isMouseFocusable() const final { return isFocusable(); }
    bool isKeyboardFocusable(KeyboardEvent*) const final { return isFocusable(); }
    void updateFocusAppearance(SelectionRestorationMode, SelectionRevealMode) final;

    bool accessKeyAction(bool) final;

    bool shouldUseInputMethod() final { return true; }
    bool matchesReadWritePseudoClass() const final { return isMutable(); }

    bool valueMissing() const final;
    bool tooShort() const final;
    bool tooLong() const final;
    bool isValidValue(StringView) const;

    bool valueMissing(StringView valueOverride) const;
    bool tooShort(StringView valueOverride, NeedsToCheckDirtyFlag) const;
    bool tooLong(StringView valueOverride, NeedsToCheckDirtyFlag) const;

    RefPtr<TextControlInnerTextElement> innerTextElementCreatingShadowSubtreeIfNeeded() final;
    RenderStyle createInnerTextStyle(const RenderStyle&) final;
    void copyNonAttributePropertiesFromElement(const Element&) final;

    bool willRespondToMouseClickEventsWithEditability(Editability) const final { return !isDisabledFormControl(); }

    RenderTextControlMultiLine* renderer() const;

    enum WrapMethod : uint8_t { NoWrap, SoftWrap, HardWrap };

    static constexpr unsigned defaultRows = 2;
    static constexpr unsigned defaultCols = 20;

    unsigned m_rows { defaultRows };
    unsigned m_cols { defaultCols };
    RefPtr<HTMLElement> m_placeholder;
    mutable String m_value;
    WrapMethod m_wrap { SoftWrap };
    mutable uint8_t m_isDirty { false }; // uint8_t for better packing on Windows
    mutable uint8_t m_wasModifiedByUser { false }; // uint8_t for better packing on Windows
};

} // namespace
