/* -*- Mode: C; tab-width: 8; indent-tabs-mode: t; c-basic-offset: 8 -*-
 * vi:set noexpandtab tabstop=8 shiftwidth=8:
 *
 * Copyright (C) 2018 Richard Hughes <richard@hughsie.com>
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "config.h"

#include "gs-prefs-dialog.h"

#include "gnome-software-private.h"
#include "gs-common.h"
#include "gs-os-release.h"
#include "gs-repo-row.h"
#include <glib/gi18n.h>

struct _GsPrefsDialog
{
	AdwPreferencesDialog	 parent_instance;
	GSettings		*settings;

	GCancellable		*cancellable;
	GsPluginLoader		*plugin_loader;
	GtkWidget		*automatic_updates_radio;
	GtkWidget		*manual_updates_radio;
	GtkLabel                *updates_info_label;
	AdwActionRow		*automatic_updates_row;
	AdwActionRow		*manual_updates_row;
	AdwActionRow		*automatic_update_notifications_row;
	AdwActionRow		*show_only_free_apps_row;
	AdwActionRow		*show_only_verified_apps_row;
};

G_DEFINE_TYPE (GsPrefsDialog, gs_prefs_dialog, ADW_TYPE_PREFERENCES_DIALOG)

static void
gs_prefs_dialog_filters_changed_cb (GsPrefsDialog *self)
{
	g_signal_emit_by_name (self->plugin_loader, "reload", 0);
}

static void
popover_show_cb (GsPrefsDialog *self)
{
    const char *label = gtk_label_get_label (self->updates_info_label);

    gtk_accessible_announce (GTK_ACCESSIBLE (self),
                             label,
                             GTK_ACCESSIBLE_ANNOUNCEMENT_PRIORITY_MEDIUM);
}

static gboolean
gs_prefs_dialog_automatic_updates_to_radio_cb (GValue *value,
					       GVariant *variant,
					       gpointer user_data)
{
	GsPrefsDialog *self = user_data;
	if (!g_variant_get_boolean (variant))
		gtk_check_button_set_active (GTK_CHECK_BUTTON (self->manual_updates_radio), TRUE);
	g_value_set_boolean (value, g_variant_get_boolean (variant));
	return TRUE;
}

static void
gs_prefs_dialog_dispose (GObject *object)
{
	GsPrefsDialog *dialog = GS_PREFS_DIALOG (object);
	g_clear_object (&dialog->plugin_loader);
	g_cancellable_cancel (dialog->cancellable);
	g_clear_object (&dialog->cancellable);
	g_clear_object (&dialog->settings);

	G_OBJECT_CLASS (gs_prefs_dialog_parent_class)->dispose (object);
}

static void
gs_prefs_dialog_init (GsPrefsDialog *dialog)
{
	gtk_widget_init_template (GTK_WIDGET (dialog));

	dialog->cancellable = g_cancellable_new ();
	dialog->settings = g_settings_new ("org.gnome.software");
	g_settings_bind (dialog->settings,
			 "download-updates-notify",
			 dialog->automatic_update_notifications_row,
			 "active",
			 G_SETTINGS_BIND_DEFAULT);
	g_settings_bind_with_mapping (dialog->settings,
				      "download-updates",
				      dialog->automatic_updates_radio,
				      "active",
				      G_SETTINGS_BIND_DEFAULT,
				      gs_prefs_dialog_automatic_updates_to_radio_cb,
				      NULL, dialog, NULL);
	g_settings_bind (dialog->settings,
			 "show-only-free-apps",
			 dialog->show_only_free_apps_row,
			 "active",
			 G_SETTINGS_BIND_DEFAULT);
	g_settings_bind (dialog->settings,
			 "show-only-verified-apps",
			 dialog->show_only_verified_apps_row,
			 "active",
			 G_SETTINGS_BIND_DEFAULT);
	g_signal_connect_object (dialog->show_only_free_apps_row, "notify::active",
				 G_CALLBACK (gs_prefs_dialog_filters_changed_cb), dialog, G_CONNECT_SWAPPED);
	g_signal_connect_object (dialog->show_only_verified_apps_row, "notify::active",
				 G_CALLBACK (gs_prefs_dialog_filters_changed_cb), dialog, G_CONNECT_SWAPPED);
}

static void
gs_prefs_dialog_class_init (GsPrefsDialogClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);
	GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

	object_class->dispose = gs_prefs_dialog_dispose;

	gtk_widget_class_set_template_from_resource (widget_class, "/org/gnome/Software/gs-prefs-dialog.ui");
	gtk_widget_class_bind_template_child (widget_class, GsPrefsDialog, automatic_updates_radio);
	gtk_widget_class_bind_template_child (widget_class, GsPrefsDialog, manual_updates_radio);
	gtk_widget_class_bind_template_child (widget_class, GsPrefsDialog, updates_info_label);
	gtk_widget_class_bind_template_child (widget_class, GsPrefsDialog, automatic_updates_row);
	gtk_widget_class_bind_template_child (widget_class, GsPrefsDialog, manual_updates_row);
	gtk_widget_class_bind_template_child (widget_class, GsPrefsDialog, automatic_update_notifications_row);
	gtk_widget_class_bind_template_child (widget_class, GsPrefsDialog, show_only_free_apps_row);
	gtk_widget_class_bind_template_child (widget_class, GsPrefsDialog, show_only_verified_apps_row);

    	gtk_widget_class_bind_template_callback (widget_class, popover_show_cb);
}

GtkWidget *
gs_prefs_dialog_new (GsPluginLoader *plugin_loader)
{
	GsPrefsDialog *dialog;
	dialog = g_object_new (GS_TYPE_PREFS_DIALOG,
			       NULL);
	dialog->plugin_loader = g_object_ref (plugin_loader);
	return GTK_WIDGET (dialog);
}
