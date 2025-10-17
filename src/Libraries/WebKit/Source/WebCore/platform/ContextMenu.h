/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#ifndef ContextMenu_h
#define ContextMenu_h

#if ENABLE(CONTEXT_MENUS)

#include <wtf/Noncopyable.h>

#include "ContextMenuItem.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ContextMenuController;

class ContextMenu {
    WTF_MAKE_TZONE_ALLOCATED(ContextMenu);
    WTF_MAKE_NONCOPYABLE(ContextMenu);
public:
    ContextMenu();

    void setItems(const Vector<ContextMenuItem>& items) { m_items = items; }
    const Vector<ContextMenuItem>& items() const { return m_items; }

    void appendItem(const ContextMenuItem& item) { m_items.append(item); } 

private:
    Vector<ContextMenuItem> m_items;
};

}

#endif // ENABLE(CONTEXT_MENUS)
#endif // ContextMenu_h
