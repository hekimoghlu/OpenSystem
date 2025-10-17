/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#include "config.h"
#include "HTMLSelectElement.h"

#if OS(WINDOWS)

#include "Element.h"
#include "KeyboardEvent.h"
#include "RenderMenuList.h"

namespace WebCore {

bool HTMLSelectElement::platformHandleKeydownEvent(KeyboardEvent* event)
{
    // Allow (Shift) F4 and (Ctrl/Shift) Alt/AltGr + Up/Down arrow to pop the menu, matching Firefox.
    bool eventShowsMenu = ((!event->altKey() && !event->ctrlKey() && event->keyIdentifier() == "F4"_s)
        || event->altKey()) && (event->keyIdentifier() == "Down"_s || event->keyIdentifier() == "Up"_s);
    if (!eventShowsMenu)
        return false;

    // Save the selection so it can be compared to the new selection when dispatching change events during setSelectedIndex,
    // which gets called from RenderMenuList::valueChanged, which gets called after the user makes a selection from the menu.
    saveLastSelection();
    if (auto* menuList = downcast<RenderMenuList>(renderer()))
        menuList->showPopup();

    int index = selectedIndex();
    ASSERT(index >= 0);
    ASSERT_WITH_SECURITY_IMPLICATION(index < static_cast<int>(listItems().size()));
    setSelectedIndex(index);
    event->setDefaultHandled();
    return true;
}

}

#endif
