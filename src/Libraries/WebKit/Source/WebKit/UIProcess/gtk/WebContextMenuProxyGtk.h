/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

#include "WebContextMenuItemGlib.h"
#include "WebContextMenuProxy.h"
#include <WebCore/GtkVersioning.h>
#include <WebCore/IntPoint.h>
#include <wtf/HashMap.h>
#include <wtf/glib/GRefPtr.h>

typedef struct _GMenu GMenu;

namespace WebKit {

class WebContextMenuItem;
class WebContextMenuItemData;
class WebPageProxy;

class WebContextMenuProxyGtk final : public WebContextMenuProxy {
public:
    static auto create(GtkWidget* widget, WebPageProxy& page, ContextMenuContextData&& context, const UserData& userData)
    {
        return adoptRef(*new WebContextMenuProxyGtk(widget, page, WTFMove(context), userData));
    }
    ~WebContextMenuProxyGtk();

    void populate(const Vector<WebContextMenuItemGlib>&);
    GtkWidget* gtkWidget() const { return m_menu; }
    static const char* widgetDismissedSignal;

private:
    WebContextMenuProxyGtk(GtkWidget*, WebPageProxy&, ContextMenuContextData&&, const UserData&);
    void show() override;
    Vector<Ref<WebContextMenuItem>> proposedItems() const override;
    void showContextMenuWithItems(Vector<Ref<WebContextMenuItem>>&&) override;
    void append(GMenu*, const WebContextMenuItemGlib&);
    GRefPtr<GMenu> buildMenu(const Vector<WebContextMenuItemGlib>&);
    void populate(const Vector<Ref<WebContextMenuItem>>&);
    Vector<WebContextMenuItemGlib> populateSubMenu(const WebContextMenuItemData&);

    GtkWidget* m_webView;
    GtkWidget* m_menu;
    HashMap<unsigned long, void*> m_signalHandlers;
    GRefPtr<GSimpleActionGroup> m_actionGroup { adoptGRef(g_simple_action_group_new()) };
};


} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
