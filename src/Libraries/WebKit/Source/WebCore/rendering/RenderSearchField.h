/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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

#include "PopupMenuClient.h"
#include "RenderTextControlSingleLine.h"
#include "SearchPopupMenu.h"

namespace WebCore {

class HTMLInputElement;

class RenderSearchField final : public RenderTextControlSingleLine, private PopupMenuClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSearchField);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSearchField);
public:
    RenderSearchField(HTMLInputElement&, RenderStyle&&);
    virtual ~RenderSearchField();

    void updateCancelButtonVisibility() const;

    void addSearchResult();

    bool popupIsVisible() const { return m_searchPopupIsVisible; }
    void showPopup();
    void hidePopup();
    WEBCORE_EXPORT std::span<const RecentSearch> recentSearches();

    // CheckedPtr interface.
    uint32_t checkedPtrCount() const final { return RenderTextControlSingleLine::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return RenderTextControlSingleLine::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { RenderTextControlSingleLine::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { RenderTextControlSingleLine::decrementCheckedPtrCount(); }

private:
    void willBeDestroyed() override;
    LayoutUnit computeControlLogicalHeight(LayoutUnit lineHeight, LayoutUnit nonContentHeight) const override;
    void updateFromElement() override;
    Visibility visibilityForCancelButton() const;
    const AtomString& autosaveName() const;

    // PopupMenuClient methods
    void valueChanged(unsigned listIndex, bool fireEvents = true) override;
    void selectionChanged(unsigned, bool) override { }
    void selectionCleared() override { }
    String itemText(unsigned listIndex) const override;
    String itemLabel(unsigned listIndex) const override;
    String itemIcon(unsigned listIndex) const override;
    String itemToolTip(unsigned) const override { return String(); }
    String itemAccessibilityText(unsigned) const override { return String(); }
    bool itemIsEnabled(unsigned listIndex) const override;
    PopupMenuStyle itemStyle(unsigned listIndex) const override;
    PopupMenuStyle menuStyle() const override;
    int clientInsetLeft() const override;
    int clientInsetRight() const override;
    LayoutUnit clientPaddingLeft() const override;
    LayoutUnit clientPaddingRight() const override;
    int listSize() const override;
    int selectedIndex() const override;
    void popupDidHide() override;
    bool itemIsSeparator(unsigned listIndex) const override;
    bool itemIsLabel(unsigned listIndex) const override;
    bool itemIsSelected(unsigned listIndex) const override;
    bool shouldPopOver() const override { return false; }
    void setTextFromItem(unsigned listIndex) override;
    FontSelector* fontSelector() const override;
    HostWindow* hostWindow() const override;
    Ref<Scrollbar> createScrollbar(ScrollableArea&, ScrollbarOrientation, ScrollbarWidth) override;
    RefPtr<SearchPopupMenu> protectedSearchPopup() const { return m_searchPopup; };

    HTMLElement* resultsButtonElement() const;
    HTMLElement* cancelButtonElement() const;

    bool m_searchPopupIsVisible;
    RefPtr<SearchPopupMenu> m_searchPopup;
    Vector<RecentSearch> m_recentSearches;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSearchField, isRenderSearchField())
