/* -*- Mode: C; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
/***************************************************************************
 * Copyright (C) 2008 Dan Williams, <dcbw@redhat.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 **************************************************************************/

#ifndef _NM_PPTP_H_
#define _NM_PPTP_H_

#if !GTK_CHECK_VERSION(4,0,0)
#define gtk_editable_set_text(editable,text)		gtk_entry_set_text(GTK_ENTRY(editable), (text))
#define gtk_editable_get_text(editable)			gtk_entry_get_text(GTK_ENTRY(editable))
#define gtk_widget_get_root(widget)			gtk_widget_get_toplevel(widget)
#define gtk_window_destroy(window)			gtk_widget_destroy(GTK_WIDGET (window))
#define gtk_check_button_get_active(button)		gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(button))
#define gtk_check_button_set_active(button, active)	gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(button), active)

typedef void GtkRoot;
#endif

#define PPTP_TYPE_PLUGIN_UI_WIDGET            (pptp_plugin_ui_widget_get_type ())
#define PPTP_PLUGIN_UI_WIDGET(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), PPTP_TYPE_PLUGIN_UI_WIDGET, PptpPluginUiWidget))
#define PPTP_PLUGIN_UI_WIDGET_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), PPTP_TYPE_PLUGIN_UI_WIDGET, PptpPluginUiWidgetClass))
#define PPTP_IS_PLUGIN_UI_WIDGET(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), PPTP_TYPE_PLUGIN_UI_WIDGET))
#define PPTP_IS_PLUGIN_UI_WIDGET_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), PPTP_TYPE_PLUGIN_UI_WIDGET))
#define PPTP_PLUGIN_UI_WIDGET_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), PPTP_TYPE_PLUGIN_UI_WIDGET, PptpPluginUiWidgetClass))

typedef struct _PptpPluginUiWidget PptpPluginUiWidget;
typedef struct _PptpPluginUiWidgetClass PptpPluginUiWidgetClass;

struct _PptpPluginUiWidget {
	GObject parent;
};

struct _PptpPluginUiWidgetClass {
	GObjectClass parent;
};

GType pptp_plugin_ui_widget_get_type (void);

NMVpnEditor *nm_vpn_plugin_ui_widget_interface_new (NMConnection *connection, GError **error);

#endif /* _NM_PPTP_H_ */

