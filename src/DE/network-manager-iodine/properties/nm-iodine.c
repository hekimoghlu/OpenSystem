/* -*- Mode: C; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
/*
 * GNOME UI dialogs for configuring iodine VPN connections
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
 * Copyright © 2012 Guido Günther <agx@sigxcpu.org>
 *
 * Based on network-manager-{openconnect,pptp}
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdlib.h>
#include <glib/gi18n-lib.h>
#include <string.h>
#include <gtk/gtk.h>

#include <NetworkManager.h>

#define IODINE_EDITOR_PLUGIN_ERROR                     NM_CONNECTION_ERROR
#define IODINE_EDITOR_PLUGIN_ERROR_INVALID_PROPERTY    NM_CONNECTION_ERROR_INVALID_PROPERTY

#include "nm-iodine-service-defines.h"
#include "nm-iodine.h"

#define IODINE_PLUGIN_NAME    _("Iodine DNS Tunnel")
#define IODINE_PLUGIN_DESC    _("Tunnel connections via DNS.")
#define IODINE_PLUGIN_SERVICE NM_DBUS_SERVICE_IODINE

#define PW_TYPE_SAVE   0
#define PW_TYPE_ASK    1
#define PW_TYPE_UNUSED 2

/************** plugin class **************/

enum {
	PROP_0,
	PROP_NAME,
	PROP_DESC,
	PROP_SERVICE
};

static void iodine_editor_plugin_interface_init (NMVpnEditorPluginInterface *iface_class);

G_DEFINE_TYPE_EXTENDED (IodineEditorPlugin, iodine_editor_plugin, G_TYPE_OBJECT, 0,
                        G_IMPLEMENT_INTERFACE (NM_TYPE_VPN_EDITOR_PLUGIN,
                                               iodine_editor_plugin_interface_init))

/************** UI widget class **************/

static void iodine_editor_interface_init (NMVpnEditorInterface *iface_class);

typedef struct {
	GtkBuilder *builder;
	GtkWidget *widget;
	GtkSizeGroup *group;
	gboolean window_added;
} IodineEditorPrivate;

G_DEFINE_TYPE_WITH_CODE (IodineEditor, iodine_editor,G_TYPE_OBJECT,
						 G_ADD_PRIVATE (IodineEditor)
						 G_IMPLEMENT_INTERFACE (NM_TYPE_VPN_EDITOR,
												iodine_editor_interface_init))

typedef enum {
	NM_IODINE_IMPORT_EXPORT_ERROR_UNKNOWN = 0,
	NM_IODINE_IMPORT_EXPORT_ERROR_NOT_IODINE,
	NM_IODINE_IMPORT_EXPORT_ERROR_BAD_DATA,
} NMIodineImportError;

#define NM_IODINE_IMPORT_EXPORT_ERROR nm_iodine_import_export_error_quark ()

static GQuark
nm_iodine_import_export_error_quark (void)
{
	static GQuark quark = 0;

	if (G_UNLIKELY (quark == 0))
		quark = g_quark_from_static_string ("nm-iodine-import-export-error-quark");
	return quark;
}

static NMConnection *
import (NMVpnEditorPlugin *iface, const char *path, GError **error)
{
	NMConnection *connection;
	NMSettingConnection *s_con;
	NMSettingVpn *s_vpn;
	NMSettingIP4Config *s_ip4;
	GKeyFile *keyfile;
	GKeyFileFlags flags;
	const char *buf;

	keyfile = g_key_file_new ();
	flags = G_KEY_FILE_KEEP_COMMENTS | G_KEY_FILE_KEEP_TRANSLATIONS;

	if (!g_key_file_load_from_file (keyfile, path, flags, NULL)) {
		g_set_error (error,
		             NM_IODINE_IMPORT_EXPORT_ERROR,
		             NM_IODINE_IMPORT_EXPORT_ERROR_NOT_IODINE,
		             "does not look like a %s VPN connection (parse failed)",
		             IODINE_PLUGIN_NAME);
		return NULL;
	}

	connection = nm_simple_connection_new ();
	s_con = NM_SETTING_CONNECTION (nm_setting_connection_new ());
	nm_connection_add_setting (connection, NM_SETTING (s_con));

	s_vpn = NM_SETTING_VPN (nm_setting_vpn_new ());
	g_object_set (s_vpn,
	              NM_SETTING_VPN_SERVICE_TYPE,
	              NM_DBUS_SERVICE_IODINE,
	              NULL);
	nm_connection_add_setting (connection, NM_SETTING (s_vpn));

	s_ip4 = NM_SETTING_IP4_CONFIG (nm_setting_ip4_config_new ());
	nm_connection_add_setting (connection, NM_SETTING (s_ip4));

	/* top level domain */
	buf = g_key_file_get_string (keyfile, "iodine", "topdomain", NULL);
	if (buf) {
		nm_setting_vpn_add_data_item (s_vpn, NM_IODINE_KEY_TOPDOMAIN, buf);
	} else {
		g_set_error (error,
		             NM_IODINE_IMPORT_EXPORT_ERROR,
		             NM_IODINE_IMPORT_EXPORT_ERROR_NOT_IODINE,
		             "does not look like a %s VPN connection "
		             "(no top level domain)",
		             IODINE_PLUGIN_NAME);
		g_object_unref (connection);
		return NULL;
	}

	/* Optional Settings */
	/* Description */
	buf = g_key_file_get_string (keyfile, "iodine", "Description", NULL);
	if (buf)
		g_object_set (s_con, NM_SETTING_CONNECTION_ID, buf, NULL);

	/* Name server */
	buf = g_key_file_get_string (keyfile, "iodine", "Nameserver", NULL);
	if (buf)
		nm_setting_vpn_add_data_item (s_vpn, NM_IODINE_KEY_NAMESERVER, buf);

	/* Fragment size */
	buf = g_key_file_get_string (keyfile, "iodine", "Fragsize", NULL);
	if (buf)
		nm_setting_vpn_add_data_item (s_vpn, NM_IODINE_KEY_FRAGSIZE, "yes");

	return connection;
}

static gboolean
export (NMVpnEditorPlugin *iface,
        const char *path,
        NMConnection *connection,
        GError **error)
{
	NMSettingVpn *s_vpn;
	const char *value;
	const char *topdomain = NULL;
	const char *nameserver = NULL;
	const char *fragsize = NULL;
	gboolean success = FALSE;
	FILE *f;

	f = fopen (path, "w");
	if (!f) {
		g_set_error (error,
		             NM_IODINE_IMPORT_EXPORT_ERROR,
		             NM_IODINE_IMPORT_EXPORT_ERROR_UNKNOWN,
		             "could not open file for writing");
		return FALSE;
	}

	s_vpn = nm_connection_get_setting_vpn (connection);

	value = nm_setting_vpn_get_data_item (s_vpn, NM_IODINE_KEY_TOPDOMAIN);
	if (value && strlen (value))
		topdomain = value;
	else {
		g_set_error (error,
		             NM_IODINE_IMPORT_EXPORT_ERROR,
		             NM_IODINE_IMPORT_EXPORT_ERROR_UNKNOWN,
		             "connection was incomplete (missing top level domain)");
		goto done;
	}

	value = nm_setting_vpn_get_data_item (s_vpn, NM_IODINE_KEY_NAMESERVER);
	if (value && strlen (value))
		nameserver = value;

	value = nm_setting_vpn_get_data_item (s_vpn, NM_IODINE_KEY_FRAGSIZE);
	if (value && strlen (value))
		fragsize = value;

	fprintf (f,
	         "[iodine]\n"
	         "Description=%s\n"
	         "Topdomain=%s\n"
	         "Nameserver=%s\n"
	         "Fragsize=%s\n",
	         /* Description */ nm_connection_get_id (connection),
	         /* Topdomain */   topdomain,
	         /* Nameserver */  nameserver,
	         /* Fragsize */    fragsize);

	success = TRUE;

done:
	fclose (f);
	return success;
}

static gboolean
check_validity (IodineEditor *self, GError **error)
{
	IodineEditorPrivate *priv = iodine_editor_get_instance_private (self);
	GtkWidget *widget;
	const char *str;

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "topdomain_entry"));
	str = gtk_entry_get_text (GTK_ENTRY (widget));
	if (!str || !strlen (str)) {
		g_set_error (error,
		             IODINE_EDITOR_PLUGIN_ERROR,
		             IODINE_EDITOR_PLUGIN_ERROR_INVALID_PROPERTY,
		             NM_IODINE_KEY_TOPDOMAIN);
		return FALSE;
	}

	return TRUE;
}

static void
stuff_changed_cb (GtkWidget *widget, gpointer user_data)
{
	g_signal_emit_by_name (IODINE_EDITOR (user_data), "changed");
}

static void
setup_password_widget (IodineEditor *self,
                       const char *entry_name,
                       NMSettingVpn *s_vpn,
                       const char *secret_name)
{
	IodineEditorPrivate *priv = iodine_editor_get_instance_private (self);

	NMSettingSecretFlags secret_flags = NM_SETTING_SECRET_FLAG_NONE;
	GtkWidget *widget;
	const char *value;

	/* Default to agent-owned for new connections */
	if (s_vpn == NULL)
		secret_flags = NM_SETTING_SECRET_FLAG_AGENT_OWNED;

	widget = (GtkWidget *) gtk_builder_get_object (priv->builder, entry_name);
	g_assert (widget);
	gtk_size_group_add_widget (priv->group, widget);

	if (s_vpn) {
		value = nm_setting_vpn_get_secret (s_vpn, secret_name);
		gtk_entry_set_text (GTK_ENTRY (widget), value ? value : "");
		nm_setting_get_secret_flags (NM_SETTING (s_vpn),
		                             secret_name,
		                             &secret_flags,
		                             NULL);
	}

	secret_flags &= ~(NM_SETTING_SECRET_FLAG_NOT_SAVED |
	                  NM_SETTING_SECRET_FLAG_NOT_REQUIRED);
	g_object_set_data (G_OBJECT (widget),
	                   "flags",
	                   GUINT_TO_POINTER (secret_flags));

	g_signal_connect (widget, "changed", G_CALLBACK (stuff_changed_cb), self);
}

static void
show_toggled_cb (GtkCheckButton *button, IodineEditor *self)
{
	IodineEditorPrivate *priv = iodine_editor_get_instance_private (self);
	GtkWidget *widget;
	gboolean visible;

	visible = gtk_toggle_button_get_active (GTK_TOGGLE_BUTTON (button));

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "password_entry"));
	g_assert (widget);
	gtk_entry_set_visibility (GTK_ENTRY (widget), visible);
}

static void
pw_type_combo_changed_cb (GtkWidget *combo, gpointer user_data)
{
	IodineEditor *self = IODINE_EDITOR (user_data);
	IodineEditorPrivate *priv = iodine_editor_get_instance_private (self);
	GtkWidget *entry;

	entry = GTK_WIDGET (gtk_builder_get_object (priv->builder, "password_entry"));
	g_assert (entry);

	/* If the user chose "Not required", desensitize and clear the correct
	 * password entry.
	 */
	switch (gtk_combo_box_get_active (GTK_COMBO_BOX (combo))) {
	case PW_TYPE_ASK:
	case PW_TYPE_UNUSED:
		gtk_entry_set_text (GTK_ENTRY (entry), "");
		gtk_widget_set_sensitive (entry, FALSE);
		break;
	default:
		gtk_widget_set_sensitive (entry, TRUE);
		break;
	}

	stuff_changed_cb (combo, self);
}

static void
init_one_pw_combo (IodineEditor *self,
                   NMSettingVpn *s_vpn,
                   const char *combo_name,
                   const char *secret_key,
                   const char *entry_name)
{
	IodineEditorPrivate *priv = iodine_editor_get_instance_private (self);
	int active = -1;
	GtkWidget *widget;
	GtkListStore *store;
	const char *value = NULL;
	guint32 default_idx = 1;
	NMSettingSecretFlags pw_flags = NM_SETTING_SECRET_FLAG_NONE;

	/* If there's already a password and the password type can't be found in
	 * the VPN settings, default to saving it.  Otherwise, always ask for it.
	 */
	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, entry_name));
	g_assert (widget);
	value = gtk_entry_get_text (GTK_ENTRY (widget));
	if (value && strlen (value))
		default_idx = 0;

	store = GTK_LIST_STORE(gtk_builder_get_object (priv->builder, "pass_type_model"));
	g_assert (store);

	if (s_vpn)
		nm_setting_get_secret_flags (NM_SETTING (s_vpn),
		                             secret_key,
		                             &pw_flags,
		                             NULL);
	if ((active < 0)
	    && !(pw_flags & NM_SETTING_SECRET_FLAG_NOT_SAVED)
	    && !(pw_flags & NM_SETTING_SECRET_FLAG_NOT_REQUIRED)) {
		active = PW_TYPE_SAVE;
	}

	if ((active < 0) && (pw_flags & NM_SETTING_SECRET_FLAG_NOT_SAVED))
		active = PW_TYPE_ASK;

	if ((active < 0) && (pw_flags & NM_SETTING_SECRET_FLAG_NOT_REQUIRED))
		active = PW_TYPE_UNUSED;

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, combo_name));
	g_assert (widget);
	gtk_combo_box_set_active (GTK_COMBO_BOX (widget),
	                          active < 0 ? default_idx : active);

	pw_type_combo_changed_cb (widget, self);
	g_signal_connect (G_OBJECT (widget),
	                  "changed",
	                  G_CALLBACK (pw_type_combo_changed_cb), self);
}

static gboolean
init_editor_plugin (IodineEditor *self,
                NMConnection *connection,
                GError **error)
{
	IodineEditorPrivate *priv = iodine_editor_get_instance_private (self);
	NMSettingVpn *s_vpn;
	GtkWidget *widget;
	const char *value;

	s_vpn = nm_connection_get_setting_vpn (connection);

	priv->group = gtk_size_group_new (GTK_SIZE_GROUP_HORIZONTAL);

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "topdomain_entry"));
	if (!widget)
		return FALSE;
	gtk_size_group_add_widget (priv->group, widget);
	if (s_vpn) {
		value = nm_setting_vpn_get_data_item (s_vpn, NM_IODINE_KEY_TOPDOMAIN);
		if (value)
			gtk_entry_set_text (GTK_ENTRY (widget), value);
	}
	g_signal_connect (G_OBJECT (widget),
	                  "changed",
	                  G_CALLBACK (stuff_changed_cb), self);

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "nameserver_entry"));
	if (!widget)
		return FALSE;
	gtk_size_group_add_widget (priv->group, widget);
	if (s_vpn) {
		value = nm_setting_vpn_get_data_item (s_vpn, NM_IODINE_KEY_NAMESERVER);
		if (value)
			gtk_entry_set_text (GTK_ENTRY (widget), value);
	}
	g_signal_connect (G_OBJECT (widget),
	                  "changed",
	                  G_CALLBACK (stuff_changed_cb), self);

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "fragsize_entry"));
	if (!widget)
		return FALSE;
	if (s_vpn) {
		value = nm_setting_vpn_get_data_item (s_vpn, NM_IODINE_KEY_FRAGSIZE);
		if (value)
			gtk_entry_set_text (GTK_ENTRY (widget), value);
	}
	g_signal_connect (G_OBJECT (widget),
	                  "changed",
	                  G_CALLBACK (stuff_changed_cb), self);

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "show_passwords_checkbutton"));
	g_signal_connect (G_OBJECT (widget), "toggled",
	                  (GCallback) show_toggled_cb,
	                  self);

	setup_password_widget (self,
	                       "password_entry",
	                       s_vpn,
	                       NM_IODINE_KEY_PASSWORD);

	init_one_pw_combo (self,
	                   s_vpn,
	                   "pass_type_combo",
	                   NM_IODINE_KEY_PASSWORD,
	                   "password_entry");
	return TRUE;
}

static GObject *
get_widget (NMVpnEditor *iface)
{
	IodineEditor *self = IODINE_EDITOR (iface);
	IodineEditorPrivate *priv = iodine_editor_get_instance_private (self);

	return G_OBJECT (priv->widget);
}

static void
save_password_and_flags (NMSettingVpn *s_vpn,
                         GtkBuilder *builder,
                         const char *entry_name,
                         const char *combo_name,
                         const char *secret_key)
{
	NMSettingSecretFlags flags = NM_SETTING_SECRET_FLAG_NONE;
	const char *password;
	GtkWidget *entry;
	GtkWidget *combo;

	/* Grab original password flags */
	entry = GTK_WIDGET (gtk_builder_get_object (builder, entry_name));
	flags = GPOINTER_TO_UINT (g_object_get_data (G_OBJECT (entry), "flags"));

	/* And set new ones based on the type combo */
	combo = GTK_WIDGET (gtk_builder_get_object (builder, combo_name));

	switch (gtk_combo_box_get_active (GTK_COMBO_BOX (combo))) {
	case PW_TYPE_SAVE:
		password = gtk_entry_get_text (GTK_ENTRY (entry));
		if (password && strlen (password))
			nm_setting_vpn_add_secret (s_vpn, secret_key, password);
		break;
	case PW_TYPE_UNUSED:
		flags |= NM_SETTING_SECRET_FLAG_NOT_REQUIRED;
		break;
	case PW_TYPE_ASK:
	default:
		flags |= NM_SETTING_SECRET_FLAG_NOT_SAVED;
		break;
	}

	/* Set new secret flags */
	nm_setting_set_secret_flags (NM_SETTING (s_vpn), secret_key, flags, NULL);
}

static gboolean
update_connection (NMVpnEditor *iface,
                   NMConnection *connection,
                   GError **error)
{
	IodineEditor *self = IODINE_EDITOR (iface);
	IodineEditorPrivate *priv = iodine_editor_get_instance_private (self);
	NMSettingVpn *s_vpn;
	GtkWidget *widget;
	char *str;

	if (!check_validity (self, error))
		return FALSE;

	s_vpn = NM_SETTING_VPN (nm_setting_vpn_new ());
	g_object_set (s_vpn, NM_SETTING_VPN_SERVICE_TYPE,
	              NM_DBUS_SERVICE_IODINE,
	              NULL);

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "topdomain_entry"));
	g_assert(widget);
	str = (char *) gtk_entry_get_text (GTK_ENTRY (widget));
	if (str && strlen (str))
		nm_setting_vpn_add_data_item (s_vpn, NM_IODINE_KEY_TOPDOMAIN, str);

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "nameserver_entry"));
	g_assert(widget);
	str = (char *) gtk_entry_get_text (GTK_ENTRY (widget));
	if (str && strlen (str))
		nm_setting_vpn_add_data_item (s_vpn, NM_IODINE_KEY_NAMESERVER, str);

	widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "fragsize_entry"));
	g_assert(widget);
	str = (char *) gtk_entry_get_text (GTK_ENTRY (widget));
	if (str && strlen (str))
		nm_setting_vpn_add_data_item (s_vpn, NM_IODINE_KEY_FRAGSIZE, str);

	/* User password and flags */
	save_password_and_flags (s_vpn,
	                         priv->builder,
	                         "password_entry",
	                         "pass_type_combo",
	                         NM_IODINE_KEY_PASSWORD);

	nm_connection_add_setting (connection, NM_SETTING (s_vpn));
	return TRUE;
}

static NMVpnEditor *
nm_vpn_editor_interface_new (NMConnection *connection, GError **error)
{
	NMVpnEditor *object;
	IodineEditorPrivate *priv;
	char *ui_file;

	if (error)
		g_return_val_if_fail (*error == NULL, NULL);

	object = g_object_new (IODINE_TYPE_EDITOR, NULL);

	if (!object) {
		g_set_error (error, IODINE_EDITOR_PLUGIN_ERROR, 0,
		             "could not create iodine object");
		return NULL;
	}

	priv = iodine_editor_get_instance_private (IODINE_EDITOR (object));
	ui_file = g_strdup_printf ("%s/%s", UIDIR, "nm-iodine-dialog.ui");
	priv->builder = gtk_builder_new ();

	gtk_builder_set_translation_domain (priv->builder, GETTEXT_PACKAGE);
	if (!gtk_builder_add_from_file (priv->builder, ui_file, error)) {
		g_warning ("Couldn't load builder file: %s",
		           error && *error ? (*error)->message : "(unknown)");
		g_clear_error (error);
		g_set_error (error, IODINE_EDITOR_PLUGIN_ERROR, 0,
		             "could not load required resources at %s", ui_file);
		g_free (ui_file);
		g_object_unref (object);
		return NULL;
	}
	g_free (ui_file);

	priv->widget = GTK_WIDGET (gtk_builder_get_object (priv->builder, "iodine-vbox"));
	if (!priv->widget) {
		g_set_error (error, IODINE_EDITOR_PLUGIN_ERROR, 0,
		             "could not load UI widget");
		g_object_unref (object);
		return NULL;
	}
	g_object_ref_sink (priv->widget);

	if (!init_editor_plugin (IODINE_EDITOR (object), connection, error)) {
		g_object_unref (object);
		return NULL;
	}

	return object;
}

static void
dispose (GObject *object)
{
	IodineEditor *self = IODINE_EDITOR (object);
	IodineEditorPrivate *priv = iodine_editor_get_instance_private (IODINE_EDITOR (self));

	if (priv->group)
		g_object_unref (priv->group);

	if (priv->widget)
		g_object_unref (priv->widget);

	if (priv->builder)
		g_object_unref (priv->builder);

	G_OBJECT_CLASS (iodine_editor_parent_class)->dispose (object);
}

static void
iodine_editor_class_init (IodineEditorClass *req_class)
{
	GObjectClass *object_class = G_OBJECT_CLASS (req_class);

	object_class->dispose = dispose;
}

static void
iodine_editor_init (IodineEditor *plugin)
{
}

static void
iodine_editor_interface_init (NMVpnEditorInterface *iface_class)
{
	/* interface implementation */
	iface_class->get_widget = get_widget;
	iface_class->update_connection = update_connection;
}

static guint32
get_capabilities (NMVpnEditorPlugin *iface)
{
	return (NM_VPN_EDITOR_PLUGIN_CAPABILITY_IMPORT |
			NM_VPN_EDITOR_PLUGIN_CAPABILITY_EXPORT);
}

static NMVpnEditor *
get_editor (NMVpnEditorPlugin *iface,
            NMConnection *connection,
            GError **error)
{
	return nm_vpn_editor_interface_new (connection, error);
}

static void
get_property (GObject *object, guint prop_id,
              GValue *value, GParamSpec *pspec)
{
	switch (prop_id) {
	case PROP_NAME:
		g_value_set_string (value, IODINE_PLUGIN_NAME);
		break;
	case PROP_DESC:
		g_value_set_string (value, IODINE_PLUGIN_DESC);
		break;
	case PROP_SERVICE:
		g_value_set_string (value, IODINE_PLUGIN_SERVICE);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
iodine_editor_plugin_class_init (IodineEditorPluginClass *req_class)
{
	GObjectClass *object_class = G_OBJECT_CLASS (req_class);

	object_class->get_property = get_property;

	g_object_class_override_property (object_class,
	                                  PROP_NAME,
	                                  NM_VPN_EDITOR_PLUGIN_NAME);

	g_object_class_override_property (object_class,
	                                  PROP_DESC,
	                                  NM_VPN_EDITOR_PLUGIN_DESCRIPTION);

	g_object_class_override_property (object_class,
	                                  PROP_SERVICE,
	                                  NM_VPN_EDITOR_PLUGIN_SERVICE);
}

static void
iodine_editor_plugin_init (IodineEditorPlugin *plugin)
{
}

static void
iodine_editor_plugin_interface_init (NMVpnEditorPluginInterface *iface_class)
{
	/* interface implementation */
	iface_class->get_editor = get_editor;
	iface_class->get_capabilities = get_capabilities;
	iface_class->import_from_file = import;
	iface_class->export_to_file = export;
}

G_MODULE_EXPORT NMVpnEditorPlugin *
nm_vpn_editor_plugin_factory (GError **error)
{
	if (error)
		g_return_val_if_fail (*error == NULL, NULL);

	return g_object_new (IODINE_TYPE_EDITOR_PLUGIN, NULL);
}
