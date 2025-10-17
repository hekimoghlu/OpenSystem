/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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
#include "WebContextMenuItemGlib.h"

#include <wtf/TZoneMallocInlines.h>

#if ENABLE(CONTEXT_MENUS)
#include "APIObject.h"
#include <gio/gio.h>

#if PLATFORM(GTK) && !USE(GTK4)
#include <gtk/gtk.h>
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebContextMenuItemGlib);

WebContextMenuItemGlib::WebContextMenuItemGlib(ContextMenuItemType type, ContextMenuAction action, const String& title, bool enabled, bool checked)
    : WebContextMenuItemData(type, action, String { title }, enabled, checked)
{
    ASSERT(type != ContextMenuItemType::Submenu);
    createActionIfNeeded();
}

WebContextMenuItemGlib::WebContextMenuItemGlib(const WebContextMenuItemData& data)
    : WebContextMenuItemData(data.type() == ContextMenuItemType::Submenu ? ContextMenuItemType::Action : data.type(), data.action(), String { data.title() }, data.enabled(), data.checked())
{
    createActionIfNeeded();
}

WebContextMenuItemGlib::WebContextMenuItemGlib(const WebContextMenuItemGlib& data, Vector<WebContextMenuItemGlib>&& submenu)
    : WebContextMenuItemData(ContextMenuItemType::Action, data.action(), String { data.title() }, data.enabled(), false)
{
    m_gAction = data.gAction();
    m_submenuItems = WTFMove(submenu);
#if PLATFORM(GTK) && !USE(GTK4)
    m_gtkAction = data.gtkAction();
#endif
}

static bool isGActionChecked(GAction* action)
{
    if (!g_action_get_state_type(action))
        return false;

    ASSERT(g_variant_type_equal(g_action_get_state_type(action), G_VARIANT_TYPE_BOOLEAN));
    GRefPtr<GVariant> state = adoptGRef(g_action_get_state(action));
    return g_variant_get_boolean(state.get());
}

WebContextMenuItemGlib::WebContextMenuItemGlib(GAction* action, const String& title, GVariant* target)
    : WebContextMenuItemData(g_action_get_state_type(action) ? ContextMenuItemType::CheckableAction : ContextMenuItemType::Action, ContextMenuItemBaseApplicationTag, String { title }, g_action_get_enabled(action), isGActionChecked(action))
    , m_gAction(action)
    , m_gActionTarget(target)
{
    createActionIfNeeded();
}

#if PLATFORM(GTK) && !USE(GTK4)
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
WebContextMenuItemGlib::WebContextMenuItemGlib(GtkAction* action)
    : WebContextMenuItemData(GTK_IS_TOGGLE_ACTION(action) ? ContextMenuItemType::CheckableAction : ContextMenuItemType::Action, ContextMenuItemBaseApplicationTag, String::fromUTF8(gtk_action_get_label(action)), gtk_action_get_sensitive(action), GTK_IS_TOGGLE_ACTION(action) ? gtk_toggle_action_get_active(GTK_TOGGLE_ACTION(action)) : false)
{
    m_gtkAction = action;
    createActionIfNeeded();
    g_object_set_data_full(G_OBJECT(m_gAction.get()), "webkit-gtk-action", g_object_ref(m_gtkAction), g_object_unref);
}
ALLOW_DEPRECATED_DECLARATIONS_END
#endif

WebContextMenuItemGlib::~WebContextMenuItemGlib()
{
}

GUniquePtr<char> WebContextMenuItemGlib::buildActionName() const
{
#if PLATFORM(GTK) && !USE(GTK4)
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    if (m_gtkAction)
        return GUniquePtr<char>(g_strdup(gtk_action_get_name(m_gtkAction)));
ALLOW_DEPRECATED_DECLARATIONS_END
#endif

    static uint64_t actionID = 0;
    return GUniquePtr<char>(g_strdup_printf("action-%" PRIu64, ++actionID));
}

void WebContextMenuItemGlib::createActionIfNeeded()
{
    if (type() == ContextMenuItemType::Separator)
        return;

    if (!m_gAction) {
        auto actionName = buildActionName();
        if (type() == ContextMenuItemType::CheckableAction)
            m_gAction = adoptGRef(G_ACTION(g_simple_action_new_stateful(actionName.get(), nullptr, g_variant_new_boolean(checked()))));
        else
            m_gAction = adoptGRef(G_ACTION(g_simple_action_new(actionName.get(), nullptr)));
        g_simple_action_set_enabled(G_SIMPLE_ACTION(m_gAction.get()), enabled());
    }

#if PLATFORM(GTK) && !USE(GTK4)
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    // Create the GtkAction for backwards compatibility only.
    if (!m_gtkAction) {
        if (type() == ContextMenuItemType::CheckableAction) {
            m_gtkAction = GTK_ACTION(gtk_toggle_action_new(g_action_get_name(m_gAction.get()), title().utf8().data(), nullptr, nullptr));
            gtk_toggle_action_set_active(GTK_TOGGLE_ACTION(m_gtkAction), checked());
        } else
            m_gtkAction = gtk_action_new(g_action_get_name(m_gAction.get()), title().utf8().data(), 0, nullptr);
        gtk_action_set_sensitive(m_gtkAction, enabled());
        g_object_set_data_full(G_OBJECT(m_gAction.get()), "webkit-gtk-action", m_gtkAction, g_object_unref);
    }

    g_signal_connect_object(m_gAction.get(), "activate", G_CALLBACK(gtk_action_activate), m_gtkAction, G_CONNECT_SWAPPED);
ALLOW_DEPRECATED_DECLARATIONS_END
#endif
}

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
