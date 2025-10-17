/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#ifndef PopupMenuClient_h
#define PopupMenuClient_h

#include "LayoutUnit.h"
#include "PopupMenuStyle.h"
#include "ScrollTypes.h"
#include <wtf/Forward.h>

namespace WebCore {

class Color;
class FontSelector;
class HostWindow;
class Scrollbar;
class ScrollableArea;

class PopupMenuClient {
public:
    virtual ~PopupMenuClient() = default;
    virtual void valueChanged(unsigned listIndex, bool fireEvents = true) = 0;
    virtual void selectionChanged(unsigned listIndex, bool fireEvents = true) = 0;
    virtual void selectionCleared() = 0;

    virtual String itemText(unsigned listIndex) const = 0;
    virtual String itemLabel(unsigned listIndex) const = 0;
    virtual String itemIcon(unsigned listIndex) const = 0;
    virtual String itemToolTip(unsigned listIndex) const = 0;
    virtual String itemAccessibilityText(unsigned listIndex) const = 0;
    virtual bool itemIsEnabled(unsigned listIndex) const = 0;
    virtual PopupMenuStyle itemStyle(unsigned listIndex) const = 0;
    virtual PopupMenuStyle menuStyle() const = 0;
    virtual int clientInsetLeft() const = 0;
    virtual int clientInsetRight() const = 0;
    virtual LayoutUnit clientPaddingLeft() const = 0;
    virtual LayoutUnit clientPaddingRight() const = 0;
    virtual int listSize() const = 0;
    virtual int selectedIndex() const = 0;
    virtual void popupDidHide() = 0;
    virtual bool itemIsSeparator(unsigned listIndex) const = 0;
    virtual bool itemIsLabel(unsigned listIndex) const = 0;
    virtual bool itemIsSelected(unsigned listIndex) const = 0;
    virtual bool shouldPopOver() const = 0;
    virtual void setTextFromItem(unsigned listIndex) = 0;

    virtual void listBoxSelectItem(int /*listIndex*/, bool /*allowMultiplySelections*/, bool /*shift*/, bool /*fireOnChangeNow*/ = true) { ASSERT_NOT_REACHED(); }
    virtual bool multiple() const
    {
        ASSERT_NOT_REACHED();
        return false;
    }

    virtual FontSelector* fontSelector() const = 0;
    virtual HostWindow* hostWindow() const = 0;

    virtual Ref<Scrollbar> createScrollbar(ScrollableArea&, ScrollbarOrientation, ScrollbarWidth) = 0;

    // CheckedPtr interface.
    virtual uint32_t checkedPtrCount() const = 0;
    virtual uint32_t checkedPtrCountWithoutThreadCheck() const = 0;
    virtual void incrementCheckedPtrCount() const = 0;
    virtual void decrementCheckedPtrCount() const = 0;
};

}

#endif
