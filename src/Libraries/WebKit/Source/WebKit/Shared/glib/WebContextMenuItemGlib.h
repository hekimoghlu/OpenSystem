/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
#include "WebContextMenuItemData.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GUniquePtr.h>

#if PLATFORM(GTK) && !USE(GTK4)
typedef struct _GtkAction GtkAction;
#endif // PLATFORM(GTK) && !USE(GTK4)
typedef struct _GAction GAction;

namespace WebKit {

class WebContextMenuItemGlib final : public WebContextMenuItemData {
    WTF_MAKE_TZONE_ALLOCATED(WebContextMenuItemGlib);
public:
    WebContextMenuItemGlib(WebCore::ContextMenuItemType, WebCore::ContextMenuAction, const String& title, bool enabled = true, bool checked = false);
    WebContextMenuItemGlib(const WebContextMenuItemData&);
    WebContextMenuItemGlib(const WebContextMenuItemGlib&, Vector<WebContextMenuItemGlib>&& submenu);
    WebContextMenuItemGlib(GAction*, const String& title, GVariant* target = nullptr);
#if PLATFORM(GTK) && !USE(GTK4)
    WebContextMenuItemGlib(GtkAction*);
#endif
    ~WebContextMenuItemGlib();

    // We don't use the SubmenuType internally, so check if we have submenu items.
    WebCore::ContextMenuItemType type() const { return m_submenuItems.isEmpty() ? WebContextMenuItemData::type() : WebCore::ContextMenuItemType::Submenu; }
    GAction* gAction() const { return m_gAction.get(); }
    GVariant* gActionTarget() const { return m_gActionTarget.get(); }
    const Vector<WebContextMenuItemGlib>& submenuItems() const { return m_submenuItems; }

#if PLATFORM(GTK) && !USE(GTK4)
    GtkAction* gtkAction() const { return m_gtkAction; }
#endif

private:
    GUniquePtr<char> buildActionName() const;
    void createActionIfNeeded();

    GRefPtr<GAction> m_gAction;
    GRefPtr<GVariant> m_gActionTarget;
    Vector<WebContextMenuItemGlib> m_submenuItems;
#if PLATFORM(GTK) && !USE(GTK4)
    GtkAction* m_gtkAction { nullptr };
#endif
};

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
