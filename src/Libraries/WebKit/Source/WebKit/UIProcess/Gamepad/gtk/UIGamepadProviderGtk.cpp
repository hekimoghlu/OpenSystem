/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#include "UIGamepadProvider.h"

#if ENABLE(GAMEPAD)

#include "WebKitWebViewBasePrivate.h"
#include "WebPageProxy.h"
#include <WebCore/GtkUtilities.h>
#include <wtf/ProcessPrivilege.h>
#include <wtf/glib/GRefPtr.h>

namespace WebKit {

using namespace WebCore;

static WebPageProxy* getWebPageProxy(GtkWidget* widget)
{
#if USE(GTK4)
    if (!widget)
        return nullptr;

    if (WEBKIT_IS_WEB_VIEW_BASE(widget))
        return gtk_widget_is_visible(widget) ? webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(widget)) : nullptr;

    for (auto* child = gtk_widget_get_first_child(widget); child; child = gtk_widget_get_next_sibling(child)) {
        if (WebPageProxy* proxy = getWebPageProxy(child))
            return proxy;
    }
#else
    if (!widget || !GTK_IS_CONTAINER(widget))
        return nullptr;

    if (WEBKIT_IS_WEB_VIEW_BASE(widget))
        return gtk_widget_is_visible(widget) ? webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(widget)) : nullptr;

    GUniquePtr<GList> children(gtk_container_get_children(GTK_CONTAINER(widget)));
    for (GList* iter = children.get(); iter; iter= g_list_next(iter)) {
        if (WebPageProxy* proxy = getWebPageProxy(GTK_WIDGET(iter->data)))
            return proxy;
    }
#endif // USE(GTK4)

    return nullptr;
}

WebPageProxy* UIGamepadProvider::platformWebPageProxyForGamepadInput()
{
    GUniquePtr<GList> toplevels(gtk_window_list_toplevels());
    for (GList* iter = toplevels.get(); iter; iter = g_list_next(iter)) {
        if (!WebCore::widgetIsOnscreenToplevelWindow(GTK_WIDGET(iter->data)))
            continue;

#if USE(GTK4)
        GtkWindow* window = GTK_WINDOW(iter->data);
        if (!gtk_window_is_active(window))
            continue;
#else
        GtkWindow* window = GTK_WINDOW(iter->data);
        if (!gtk_window_has_toplevel_focus(window))
            continue;
#endif // USE(GTK4)

        if (WebPageProxy* proxy = getWebPageProxy(GTK_WIDGET(window)))
            return proxy;
    }
    return nullptr;
}

}

#endif // ENABLE(GAMEPAD)
