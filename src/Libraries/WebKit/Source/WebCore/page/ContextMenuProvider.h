/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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

#if ENABLE(CONTEXT_MENUS)

#include "ContextMenuContext.h"
#include "ContextMenuItem.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class ContextMenu;

class ContextMenuProvider : public RefCounted<ContextMenuProvider> {
public:
    virtual ~ContextMenuProvider() { };

    virtual void populateContextMenu(ContextMenu*) = 0;
    virtual void didDismissContextMenu() { }
    virtual void contextMenuItemSelected(ContextMenuAction, const String& title) = 0;
    virtual void contextMenuCleared() = 0;
    virtual ContextMenuContext::Type contextMenuContextType() { return ContextMenuContext::Type::ContextMenu; };
};

} // namespace WebCore

#endif // ENABLE(CONTEXT_MENUS)
