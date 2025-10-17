/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

#include "HTMLFormControlElement.h"
#include "HTMLOptionElement.h"
#include "TypeAhead.h"
#include <wtf/CompletionHandler.h>

namespace WebCore {

class HTMLOptionsCollection;

class HTMLSelectElement : public HTMLFormControlElement, private TypeAheadDataSource {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLSelectElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLSelectElement);
public:
    static Ref<HTMLSelectElement> create(const QualifiedName&, Document&, HTMLFormElement*);
    static Ref<HTMLSelectElement> create(Document&);

    WEBCORE_EXPORT int selectedIndex() const;
    WEBCORE_EXPORT void setSelectedIndex(int);

    WEBCORE_EXPORT void optionSelectedByUser(int index, bool dispatchChangeEvent, bool allowMultipleSelection = false);

    String validationMessage() const final;
    bool valueMissing() const final;

    WEBCORE_EXPORT unsigned length() const;

    unsigned size() const { return m_size; }
    bool multiple() const { return m_multiple; }

    bool usesMenuList() const;

    using OptionOrOptGroupElement = std::variant<RefPtr<HTMLOptionElement>, RefPtr<HTMLOptGroupElement>>;
    using HTMLElementOrInt = std::variant<RefPtr<HTMLElement>, int>;
    WEBCORE_EXPORT ExceptionOr<void> add(const OptionOrOptGroupElement&, const std::optional<HTMLElementOrInt>& before);

    using Node::remove;
    WEBCORE_EXPORT void remove(int);

    WEBCORE_EXPORT String value() const;
    WEBCORE_EXPORT void setValue(const String&);

    WEBCORE_EXPORT Ref<HTMLOptionsCollection> options();
    Ref<HTMLCollection> selectedOptions();

    void optionElementChildrenChanged();

    void setRecalcListItems();
    void invalidateSelectedItems();
    void updateListItemSelectedStates(AllowStyleInvalidation = AllowStyleInvalidation::Yes);

    WEBCORE_EXPORT const Vector<WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData>>& listItems() const;

    void accessKeySetSelectedIndex(int);

    WEBCORE_EXPORT void setSize(unsigned);

    // Called by the bindings for the unnamed index-setter.
    ExceptionOr<void> setItem(unsigned index, HTMLOptionElement*);
    ExceptionOr<void> setLength(unsigned);

    ExceptionOr<void> showPicker();

    WEBCORE_EXPORT HTMLOptionElement* namedItem(const AtomString& name);
    WEBCORE_EXPORT HTMLOptionElement* item(unsigned index);
    bool isSupportedPropertyIndex(unsigned index);

    void scrollToSelection();

    void listBoxSelectItem(int listIndex, bool allowMultiplySelections, bool shift, bool fireOnChangeNow = true);

    bool canSelectAll() const;
    void selectAll();
    int listToOptionIndex(int listIndex) const;
    void listBoxOnChange();
    int optionToListIndex(int optionIndex) const;
    int activeSelectionStartListIndex() const;
    int activeSelectionEndListIndex() const;
    void setActiveSelectionAnchorIndex(int);
    void setActiveSelectionEndIndex(int);
    void updateListBoxSelection(bool deselectOtherOptions);

    // For use in the implementation of HTMLOptionElement.
    void optionSelectionStateChanged(HTMLOptionElement&, bool optionIsSelected);
    bool allowsNonContiguousSelection() const { return m_allowsNonContiguousSelection; };

    CompletionHandlerCallingScope optionToSelectFromChildChangeScope(const ChildChange&, HTMLOptGroupElement* parentOptGroup = nullptr);

    bool canContainRangeEndPoint() const override { return false; }
    bool shouldSaveAndRestoreFormControlState() const final { return true; }

    bool isDevolvableWidget() const override { return true; }

protected:
    HTMLSelectElement(const QualifiedName&, Document&, HTMLFormElement*);

private:
    const AtomString& formControlType() const final;

    int defaultTabIndex() const final;
    bool isKeyboardFocusable(KeyboardEvent*) const final;
    bool isMouseFocusable() const final;

    void dispatchFocusEvent(RefPtr<Element>&& oldFocusedElement, const FocusOptions&) final;
    void dispatchBlurEvent(RefPtr<Element>&& newFocusedElement) final;
    
    bool canStartSelection() const final { return false; }

    bool isEnumeratable() const final { return true; }
    bool isLabelable() const final { return true; }

    bool isInteractiveContent() const final { return true; }

    FormControlState saveFormControlState() const final;
    void restoreFormControlState(const FormControlState&) final;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;

    bool childShouldCreateRenderer(const Node&) const final;
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    bool appendFormData(DOMFormData&) final;

    void reset() final;

    void defaultEventHandler(Event&) final;
    bool willRespondToMouseClickEventsWithEditability(Editability) const final;

    void dispatchChangeEventForMenuList();

    void didRecalcStyle(Style::Change) final;

    void recalcListItems(bool updateSelectedStates = true, AllowStyleInvalidation = AllowStyleInvalidation::Yes) const;

    void typeAheadFind(KeyboardEvent&);
    void saveLastSelection();

    bool isOptionalFormControl() const final { return !isRequiredFormControl(); }
    bool isRequiredFormControl() const final;

    bool hasPlaceholderLabelOption() const;

    enum SelectOptionFlag {
        DeselectOtherOptions = 1 << 0,
        DispatchChangeEvent = 1 << 1,
        UserDriven = 1 << 2,
    };
    typedef unsigned SelectOptionFlags;
    void selectOption(int optionIndex, SelectOptionFlags = 0);
    void deselectItemsWithoutValidation(HTMLElement* elementToExclude = nullptr);
    void parseMultipleAttribute(const AtomString&);
    int lastSelectedListIndex() const;
    void updateSelectedState(int listIndex, bool multi, bool shift);
    void menuListDefaultEventHandler(Event&);
    bool platformHandleKeydownEvent(KeyboardEvent*);
    void listBoxDefaultEventHandler(Event&);
    void setOptionsChangedOnRenderer();
    size_t searchOptionsForValue(const String&, size_t listIndexStart, size_t listIndexEnd) const;

    enum SkipDirection { SkipBackwards = -1, SkipForwards = 1 };
    int nextValidIndex(int listIndex, SkipDirection, int skip) const;
    int nextSelectableListIndex(int startIndex) const;
    int previousSelectableListIndex(int startIndex) const;
    int firstSelectableListIndex() const;
    int lastSelectableListIndex() const;
    int nextSelectableListIndexPageAway(int startIndex, SkipDirection) const;

    void childrenChanged(const ChildChange&) final;

    // TypeAheadDataSource functions.
    int indexOfSelectedOption() const final;
    int optionCount() const final;
    String optionAtIndex(int index) const final;


    // m_listItems contains HTMLOptionElement, HTMLOptGroupElement, and HTMLHRElement objects.
    mutable Vector<WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData>> m_listItems;
    Vector<bool> m_lastOnChangeSelection;
    Vector<bool> m_cachedStateForActiveSelection;
    TypeAhead m_typeAhead;
    unsigned m_size;
    int m_lastOnChangeIndex;
    int m_activeSelectionAnchorIndex;
    int m_activeSelectionEndIndex;
    bool m_isProcessingUserDrivenChange;
    bool m_multiple;
    bool m_activeSelectionState;
    bool m_allowsNonContiguousSelection;
    mutable bool m_shouldRecalcListItems;
};

} // namespace
