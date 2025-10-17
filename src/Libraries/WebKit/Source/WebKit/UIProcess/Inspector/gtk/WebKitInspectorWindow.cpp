/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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
#include "WebKitInspectorWindow.h"

#include "WebInspectorUIProxy.h"
#include <glib/gi18n-lib.h>
#include <wtf/glib/GUniquePtr.h>

using namespace WebKit;

struct _WebKitInspectorWindow {
    GtkWindow parent;

    GtkWidget* headerBar;

#if USE(GTK4)
    GtkWidget* subtitleLabel;
#endif
};

struct _WebKitInspectorWindowClass {
    GtkWindowClass parent;
};

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK port
G_DEFINE_TYPE(WebKitInspectorWindow, webkit_inspector_window, GTK_TYPE_WINDOW)
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

static void webkit_inspector_window_class_init(WebKitInspectorWindowClass*)
{
}

static void webkit_inspector_window_init(WebKitInspectorWindow* window)
{
    window->headerBar = gtk_header_bar_new();

#if USE(GTK4)
    GtkWidget* box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_valign(box, GTK_ALIGN_CENTER);

    GtkWidget* titleLabel = gtk_label_new(_("Web Inspector"));
    gtk_widget_set_halign(titleLabel, GTK_ALIGN_CENTER);
    gtk_label_set_single_line_mode(GTK_LABEL(titleLabel), TRUE);
    gtk_label_set_ellipsize(GTK_LABEL(titleLabel), PANGO_ELLIPSIZE_END);
    gtk_widget_add_css_class(titleLabel, "title");
    gtk_widget_set_parent(titleLabel, box);

    window->subtitleLabel = gtk_label_new(nullptr);
    gtk_widget_set_halign(window->subtitleLabel, GTK_ALIGN_CENTER);
    gtk_label_set_single_line_mode(GTK_LABEL(window->subtitleLabel), TRUE);
    gtk_label_set_ellipsize(GTK_LABEL(window->subtitleLabel), PANGO_ELLIPSIZE_END);
    gtk_widget_add_css_class(window->subtitleLabel, "subtitle");
    gtk_widget_set_parent(window->subtitleLabel, box);
    gtk_widget_hide(window->subtitleLabel);

    gtk_header_bar_set_title_widget(GTK_HEADER_BAR(window->headerBar), box);
    gtk_header_bar_set_show_title_buttons(GTK_HEADER_BAR(window->headerBar), TRUE);
#else
    gtk_header_bar_set_title(GTK_HEADER_BAR(window->headerBar), _("Web Inspector"));
    gtk_header_bar_set_show_close_button(GTK_HEADER_BAR(window->headerBar), TRUE);
    gtk_widget_show(window->headerBar);
#endif // USE(GTK4)

    gtk_window_set_titlebar(GTK_WINDOW(window), window->headerBar);
}

GtkWidget* webkitInspectorWindowNew()
{
    return GTK_WIDGET(g_object_new(WEBKIT_TYPE_INSPECTOR_WINDOW,
#if !USE(GTK4)
        "type", GTK_WINDOW_TOPLEVEL,
#endif
        "default-width", WebInspectorUIProxy::initialWindowWidth, "default-height", WebInspectorUIProxy::initialWindowHeight, nullptr));
}

void webkitInspectorWindowSetSubtitle(WebKitInspectorWindow* window, const char* subtitle)
{
    g_return_if_fail(WEBKIT_IS_INSPECTOR_WINDOW(window));

#if USE(GTK4)
    if (subtitle) {
        gtk_label_set_text(GTK_LABEL(window->subtitleLabel), subtitle);
        gtk_widget_show(window->subtitleLabel);
    } else
        gtk_widget_hide(window->subtitleLabel);
#else
    gtk_header_bar_set_subtitle(GTK_HEADER_BAR(window->headerBar), subtitle);
#endif // USE(GTK4)
}
