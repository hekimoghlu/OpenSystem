/* -*- Mode: C; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
/*
 * NetworkManager iodine VPN connections
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
 * Copyright © 2012,2017 Guido Günther <agx@sigxcpu.org>
 *
 * Based on network-manager-{openconnect,pptp}
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/if_tun.h>
#include <net/if.h>
#include <pwd.h>
#include <grp.h>
#include <glib/gi18n.h>
#include <arpa/inet.h>

#include <NetworkManager.h>
#include <nm-vpn-service-plugin.h>
#include "nm-iodine-service.h"
#include "nm-utils.h"

#define NM_IODINE_USER "nm-iodine"
#define NM_IODINE_RUNDIR LOCALSTATEDIR "/run/" NM_IODINE_USER

typedef struct {
	GPid pid;
	NMVpnPluginFailure failure;
	GVariantBuilder ip4config;
} NMIodinePluginPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (NMIodinePlugin, nm_iodine_plugin, NM_TYPE_VPN_SERVICE_PLUGIN)

static const char *iodine_binary_paths[] =
{
	"/usr/bin/iodine",
	"/usr/sbin/iodine",
	"/usr/local/bin/iodine",
	"/usr/local/sbin/iodine",
	"/opt/bin/iodine",
	"/opt/sbin/iodine",
	NULL
};

typedef struct {
	const char *name;
	GType type;
	gint int_min;
	gint int_max;
} ValidProperty;

static ValidProperty valid_properties[] = {
	{ NM_IODINE_KEY_TOPDOMAIN,  G_TYPE_STRING, 0, 0 },
	{ NM_IODINE_KEY_NAMESERVER, G_TYPE_STRING, 0, 0 },
	{ NM_IODINE_KEY_FRAGSIZE,   G_TYPE_STRING, 0, 0 },
	{ NULL,                     G_TYPE_NONE, 0, 0 }
};

static ValidProperty valid_secrets[] = {
	{ NM_IODINE_KEY_PASSWORD, G_TYPE_STRING, 0, 0 },
	{ NULL,                   G_TYPE_NONE, 0, 0 }
};

typedef struct ValidateInfo {
	ValidProperty *table;
	GError **error;
	gboolean have_items;
} ValidateInfo;

static void
validate_one_property (const char *key, const char *value, gpointer user_data)
{
	ValidateInfo *info = (ValidateInfo *) user_data;
	int i;

	if (*(info->error))
		return;

	info->have_items = TRUE;

	/* 'name' is the setting name; always allowed but unused */
	if (!strcmp (key, NM_SETTING_NAME))
		return;

	for (i = 0; info->table[i].name; i++) {
		ValidProperty prop = info->table[i];
		long int tmp;

		if (strcmp (prop.name, key))
			continue;

		switch (prop.type) {
		case G_TYPE_STRING:
			return; /* valid */
		case G_TYPE_INT:
			errno = 0;
			tmp = strtol (value, NULL, 10);
			if (errno == 0 && tmp >= prop.int_min && tmp <= prop.int_max)
				return; /* valid */

			g_set_error (info->error,
			             NM_VPN_PLUGIN_ERROR,
			             NM_VPN_PLUGIN_ERROR_BAD_ARGUMENTS,
			             _("invalid integer property “%s” or out of range "
			               "[%d -> %d]"),
			             key, prop.int_min, prop.int_max);
			break;
		case G_TYPE_BOOLEAN:
			if (!strcmp (value, "yes") || !strcmp (value, "no"))
				return; /* valid */

			g_set_error (info->error,
			             NM_VPN_PLUGIN_ERROR,
			             NM_VPN_PLUGIN_ERROR_BAD_ARGUMENTS,
			             _("invalid boolean property “%s” (not yes or no)"),
			             key);
			break;
		default:
			g_set_error (info->error,
			             NM_VPN_PLUGIN_ERROR,
			             NM_VPN_PLUGIN_ERROR_BAD_ARGUMENTS,
			             _("unhandled property “%s” type %s"),
			             key, g_type_name (prop.type));
			break;
		}
	}

	/* Did not find the property from valid_properties or the type did not
	   match */
	if (!info->table[i].name && strncmp(key, "form:", 5)) {
		g_warning ("property '%s' unknown", key);
		if (0)
		g_set_error (info->error,
		             NM_VPN_PLUGIN_ERROR,
		             NM_VPN_PLUGIN_ERROR_BAD_ARGUMENTS,
		             _("property “%s” invalid or not supported"),
		             key);
	}
}

static gboolean
nm_iodine_properties_validate (NMSettingVpn *s_vpn, GError **error)
{
	ValidateInfo info = { &valid_properties[0], error, FALSE };

	nm_setting_vpn_foreach_data_item (s_vpn, validate_one_property, &info);
	if (!info.have_items) {
		g_set_error (error,
		             NM_VPN_PLUGIN_ERROR,
		             NM_VPN_PLUGIN_ERROR_BAD_ARGUMENTS,
		             "%s",
		             _("No VPN configuration options."));
		return FALSE;
	}

	return *error ? FALSE : TRUE;
}


static gboolean
nm_iodine_secrets_validate (NMSettingVpn *s_vpn, GError **error)
{
	ValidateInfo info = { &valid_secrets[0], error, FALSE };

	nm_setting_vpn_foreach_secret (s_vpn, validate_one_property, &info);
	if (!info.have_items) {
		g_set_error (error,
		             NM_VPN_PLUGIN_ERROR,
		             NM_VPN_PLUGIN_ERROR_BAD_ARGUMENTS,
		             "%s",
		             _("No VPN secrets!"));
		return FALSE;
	}

	return *error ? FALSE : TRUE;
}

static GVariant *
str_to_gvariant (const char *str, gboolean try_convert)
{
	/* Empty */
	if (!str || strlen (str) < 1)
		return NULL;

	if (!g_utf8_validate (str, -1, NULL)) {
		if (try_convert && !(str = g_convert (str,
		                                      -1,
		                                      "ISO-8859-1",
		                                      "UTF-8",
		                                      NULL,
		                                      NULL,
		                                      NULL)))
			str = g_convert (str, -1, "C", "UTF-8", NULL, NULL, NULL);
		if (!str)
			/* Invalid */
			return NULL;
	}

	return g_variant_new_string (str);
}

static GVariant *
addr4_to_gvariant (const char *str)
{
	struct in_addr	temp_addr;

	/* Empty */
	if (!str || strlen (str) < 1)
		return NULL;

	if (inet_pton (AF_INET, str, &temp_addr) <= 0)
		return NULL;

	return g_variant_new_uint32 (temp_addr.s_addr);
}

static gint
iodine_parse_stderr_line (NMVpnServicePlugin *plugin,
                          const char* line)
{
	NMIodinePluginPrivate *priv = nm_iodine_plugin_get_instance_private (NM_IODINE_PLUGIN(plugin));
	gchar **split = NULL;
	GVariant *val;
	long int mtu;
	gint len;
	gint ret = 1;

	if (g_str_has_prefix(line, "Bad password")) {
		g_message ("Login failure");
		priv->failure = NM_VPN_PLUGIN_FAILURE_LOGIN_FAILED;
		ret = -1;
		goto out;
	}

	split = g_strsplit (line, " ", 0);
	len = g_strv_length (split);
	if (len < 2)
		goto out;

	if (g_str_has_prefix(line, "Server tunnel IP is ")) {
		g_message("PTP address: %s", split[len-1]);
		val = addr4_to_gvariant (split[len-1]);
		if (val) {
			g_variant_builder_add (&priv->ip4config, "{sv}",
			                       NM_VPN_PLUGIN_IP4_CONFIG_PTP,
			                       val);
		} else {
			g_warning("Could not set PTP");
			return -1;
		}
		val = addr4_to_gvariant (split[len-1]);
		if (val) {
			g_variant_builder_add (&priv->ip4config, "{sv}",
			                       NM_VPN_PLUGIN_IP4_CONFIG_INT_GATEWAY,
			                       val);
		} else {
			g_warning("Could not set internal gateway");
			return -1;
		}
	} else if (g_str_has_prefix(line, "Sending DNS queries for ")) {
		g_message("External gw: %s", split[len-1]);
		val = addr4_to_gvariant (split[len-1]);
		if (val) {
			g_variant_builder_add (&priv->ip4config, "{sv}",
			                       NM_VPN_PLUGIN_IP4_CONFIG_EXT_GATEWAY,
			                       val);
		} else {
			g_warning("Could not set external gateway");
			return -1;
		}
	} else if (g_str_has_prefix(line, "Sending raw traffic directly to ")) {
		/* If the DNS server is directly reachable we need to set it
		   as external gateway overwriting the above values */
		g_message("Overwrite ext. gw.  address: %s", split[len-1]);
		val = addr4_to_gvariant (split[len-1]);
		if (val) {
			g_variant_builder_add (&priv->ip4config, "{sv}",
			                       NM_VPN_PLUGIN_IP4_CONFIG_EXT_GATEWAY,
			                       val);
		} else {
			g_warning("Could not set external gateway");
			return -1;
		}
	} else if (g_str_has_prefix(line, "Setting IP of dns")) {
		g_message("Address: %s", split[len-1]);
		val = addr4_to_gvariant (split[len-1]);
		if (val) {
			g_variant_builder_add (&priv->ip4config, "{sv}",
			                       NM_VPN_PLUGIN_IP4_CONFIG_ADDRESS,
			                       val);
		} else {
			g_warning("Could not set address");
			return -1;
		}
	} else if (g_str_has_prefix(line, "Setting MTU of ")) {
		errno = 0;
		mtu = strtol (split[len-1], NULL, 10);
		if (errno || mtu < 0 || mtu > 20000) {
			g_warning("Ignoring invalid tunnel MTU '%s'", split[len-1]);
		} else {
			g_message("MTU: %s", split[len-1]);
			val = g_variant_new_uint32 ((guint32) mtu);
			if (val) {
				g_variant_builder_add (&priv->ip4config, "{sv}",
									   NM_VPN_PLUGIN_IP4_CONFIG_MTU,
									   val);
			} else {
				g_warning("Could not set MTU");
				return -1;
			}
		}
	} else if (g_str_has_prefix(line, "Opened dns")) {
		g_message("Interface: %s", split[len-1]);
		val = str_to_gvariant (split[len-1], FALSE);
		if (val) {
			g_variant_builder_add (&priv->ip4config, "{sv}",
			                       NM_VPN_PLUGIN_IP4_CONFIG_TUNDEV,
			                       val);
		} else {
			g_warning("Could not set tundev");
			return -1;
		}
	} else if (g_str_has_prefix(line,
		                    "Connection setup complete, "
		                    "transmitting data.")) {
		val = g_variant_new_uint32 (27);
		g_variant_builder_add (&priv->ip4config, "{sv}",
		                       NM_VPN_PLUGIN_IP4_CONFIG_PREFIX,
		                       val);
		ret = 0; /* success */
	} else
		g_message("Unparsed: %s", line);

out:
	g_strfreev(split);
	return ret;
}


static gboolean
iodine_stderr_cb (GIOChannel *source, GIOCondition condition, gpointer plugin)
{
	GIOStatus status;
	GError *err = NULL;
	gchar *line;
	gint ret, l;
	NMIodinePluginPrivate *priv = nm_iodine_plugin_get_instance_private (NM_IODINE_PLUGIN(plugin));

	status = g_io_channel_read_line (source, &line, NULL, NULL, &err);
	switch (status) {
	case G_IO_STATUS_NORMAL:
		break;
	case G_IO_STATUS_EOF:
		return FALSE;
	default:
		g_warning ("Fetching data failed: %s",
				   (err && err->message) ? err->message : "unknown error");
		return FALSE;
	}

	l = strlen(line);
	if (l)
		line[l-1] = '\0';

	ret = iodine_parse_stderr_line(plugin, line);
	if (!ret) {
		g_message("Parsing done, sending IP4 config");
		nm_vpn_service_plugin_set_ip4_config(plugin,
		                                     g_variant_builder_end (&priv->ip4config));
	} else if (ret < 0) {
		nm_vpn_service_plugin_failure (NM_VPN_SERVICE_PLUGIN (plugin),
									   NM_VPN_PLUGIN_FAILURE_BAD_IP_CONFIG);
		g_variant_builder_end (&priv->ip4config);
		return FALSE;
	}
	g_free (line);
	return TRUE;
}


static void
iodine_watch_cb (GPid pid, gint status, gpointer user_data)
{
	NMIodinePlugin *plugin = NM_IODINE_PLUGIN (user_data);
	NMIodinePluginPrivate *priv = nm_iodine_plugin_get_instance_private (NM_IODINE_PLUGIN(plugin));
	guint error = 0;

	if (WIFEXITED (status)) {
		error = WEXITSTATUS (status);
		if (error != 0)
			g_warning ("iodine exited with error code %d", error);
	}
	else if (WIFSTOPPED (status))
		g_warning ("iodine stopped unexpectedly with signal %d",
				   WSTOPSIG (status));
	else if (WIFSIGNALED (status))
		g_warning ("iodine died with signal %d", WTERMSIG (status));
	else
		g_warning ("iodine died from an unknown cause");

	/* Reap child if needed. */
	waitpid (priv->pid, NULL, WNOHANG);
	priv->pid = 0;

	if (priv->failure >= 0) {
		nm_vpn_service_plugin_failure (NM_VPN_SERVICE_PLUGIN (plugin), priv->failure);
	} else if (error) {
		nm_vpn_service_plugin_failure (NM_VPN_SERVICE_PLUGIN (plugin), NM_VPN_PLUGIN_FAILURE_CONNECT_FAILED);
	} else {
		nm_vpn_service_plugin_disconnect (NM_VPN_SERVICE_PLUGIN (plugin), NULL);
	}

	g_variant_builder_clear(&priv->ip4config);
	priv->failure = -1;
}

static gboolean
has_user(const char* user)
{
	return (getpwnam(user) == NULL) ? FALSE : TRUE;
}


static void
send_password(gint fd, NMSettingVpn *s_vpn)
{
	const char *passwd;
	ssize_t ret;

	passwd = nm_setting_vpn_get_secret (s_vpn, NM_IODINE_KEY_PASSWORD);
	/* Don't send an empty password since this makes iodine block */
	if (!passwd || !strlen(passwd))
		passwd = "<none>";

	ret = write (fd, passwd, strlen(passwd));
	if (ret < 0)
		g_warning("Password write failed");
	ret = write (fd, "\n", 1);
	if (ret < 0)
		g_warning("Password write failed");
}


static gint
nm_iodine_start_iodine_binary (NMIodinePlugin *plugin,
                               NMSettingVpn *s_vpn,
                               GError **error)
{
	GPid pid;
	const char **iodine_binary = NULL;
	GPtrArray *iodine_argv;
	GSource *iodine_watch;
	GIOChannel *stderr_channel;
	gint stdin_fd, stderr_fd;
	const char *props_topdomain, *props_fragsize, *props_nameserver;
	NMIodinePluginPrivate *priv = nm_iodine_plugin_get_instance_private (NM_IODINE_PLUGIN(plugin));

	/* Find iodine */
	iodine_binary = iodine_binary_paths;
	while (*iodine_binary != NULL) {
		if (g_file_test (*iodine_binary, G_FILE_TEST_EXISTS))
			break;
		iodine_binary++;
	}

	if (!*iodine_binary) {
		g_set_error (error,
		             NM_VPN_PLUGIN_ERROR,
		             NM_VPN_PLUGIN_ERROR_LAUNCH_FAILED,
		             "%s",
		             _("Could not find iodine binary."));
		return -1;
	}

	props_fragsize = nm_setting_vpn_get_data_item (s_vpn, NM_IODINE_KEY_FRAGSIZE);
	props_nameserver = nm_setting_vpn_get_data_item (s_vpn, NM_IODINE_KEY_NAMESERVER);
	props_topdomain = nm_setting_vpn_get_data_item (s_vpn, NM_IODINE_KEY_TOPDOMAIN);
	iodine_argv = g_ptr_array_new ();
	g_ptr_array_add (iodine_argv, (gpointer) (*iodine_binary));
	/* Run in foreground */
	g_ptr_array_add (iodine_argv, (gpointer) "-f");

	if (props_fragsize && strlen(props_fragsize)) {
		g_ptr_array_add (iodine_argv, (gpointer) "-m");
		g_ptr_array_add (iodine_argv, (gpointer) props_fragsize);
	}

	if (has_user(NM_IODINE_USER)) {
		g_ptr_array_add (iodine_argv, (gpointer) "-u");
		g_ptr_array_add (iodine_argv, (gpointer) NM_IODINE_USER);
	} else
		g_warning("Running as root user");

	if (!g_mkdir_with_parents(NM_IODINE_RUNDIR, 700)) {
		g_ptr_array_add (iodine_argv, (gpointer) "-t");
		g_ptr_array_add (iodine_argv, (gpointer) NM_IODINE_RUNDIR);
	} else
		g_warning("Not running chrooted");

	if (props_nameserver && strlen(props_nameserver))
		g_ptr_array_add (iodine_argv, (gpointer) props_nameserver);

	if (props_topdomain && strlen(props_topdomain))
		g_ptr_array_add (iodine_argv, (gpointer) props_topdomain);

	g_ptr_array_add (iodine_argv, NULL);

	if (!g_spawn_async_with_pipes (NULL, (char **) iodine_argv->pdata, NULL,
	                               G_SPAWN_DO_NOT_REAP_CHILD,
	                               NULL, NULL,
	                               &pid, &stdin_fd, NULL, &stderr_fd, error)) {
		g_ptr_array_free (iodine_argv, TRUE);
		g_warning ("iodine failed to start. error: '%s'", (*error)->message);
		return -1;
	}
	g_ptr_array_free (iodine_argv, TRUE);

	g_message ("iodine started with pid %d", pid);

	send_password (stdin_fd, s_vpn);
	close (stdin_fd);

	stderr_channel = g_io_channel_unix_new (stderr_fd);
	g_io_add_watch(stderr_channel,
	               G_IO_IN | G_IO_HUP,
	               iodine_stderr_cb,
	               plugin);

	priv->pid = pid;
	iodine_watch = g_child_watch_source_new (pid);
	g_source_set_callback (iodine_watch,
	                       (GSourceFunc) iodine_watch_cb,
	                       plugin,
	                       NULL);
	g_source_attach (iodine_watch, NULL);
	g_source_unref (iodine_watch);

	return 0;
}

static gboolean
real_connect (NMVpnServicePlugin *plugin,
              NMConnection *connection,
              GError **error)
{
	NMSettingVpn *s_vpn;
	NMIodinePluginPrivate *priv = nm_iodine_plugin_get_instance_private (NM_IODINE_PLUGIN(plugin));
	gint ret = -1;

	g_variant_builder_init(&priv->ip4config, G_VARIANT_TYPE_VARDICT);

	s_vpn = nm_connection_get_setting_vpn (connection);
	g_assert (s_vpn);
	if (!nm_iodine_properties_validate (s_vpn, error))
		goto out;

	if (!nm_iodine_secrets_validate (s_vpn, error))
		goto out;

	ret = nm_iodine_start_iodine_binary (NM_IODINE_PLUGIN (plugin), s_vpn, error);
	if (!ret)
		return TRUE;

 out:
	g_variant_builder_clear(&priv->ip4config);
	return FALSE;
}

static gboolean
real_need_secrets (NMVpnServicePlugin *plugin,
                   NMConnection *connection,
                   const char **setting_name,
                   GError **error)
{
	NMSettingVpn *s_vpn;

	g_return_val_if_fail (NM_IS_VPN_SERVICE_PLUGIN (plugin), FALSE);
	g_return_val_if_fail (NM_IS_CONNECTION (connection), FALSE);

	s_vpn = nm_connection_get_setting_vpn (connection);
	if (!s_vpn) {
		g_set_error (error,
		             NM_VPN_PLUGIN_ERROR,
		             NM_VPN_PLUGIN_ERROR_INVALID_CONNECTION,
		             "%s",
		             "Could not process the request because the VPN "
		             "connection settings were invalid.");
		return FALSE;
	}

	if (!nm_setting_vpn_get_secret (s_vpn, NM_IODINE_KEY_PASSWORD)) {
		*setting_name = NM_SETTING_VPN_SETTING_NAME;
		return TRUE;
	}

	return FALSE;
}

static gboolean
ensure_killed (gpointer data)
{
	int pid = GPOINTER_TO_INT (data);

	if (kill (pid, 0) == 0)
		kill (pid, SIGKILL);

	return FALSE;
}

static gboolean
real_disconnect (NMVpnServicePlugin *plugin, GError **err)
{
	NMIodinePluginPrivate *priv = nm_iodine_plugin_get_instance_private (NM_IODINE_PLUGIN(plugin));

	if (priv->pid) {
		if (kill (priv->pid, SIGTERM) == 0)
			g_timeout_add (2000, ensure_killed, GINT_TO_POINTER (priv->pid));
		else
			kill (priv->pid, SIGKILL);

		g_message ("Terminated iodine daemon with PID %d.", priv->pid);
		priv->pid = 0;
	}

	return TRUE;
}

static void
nm_iodine_plugin_init (NMIodinePlugin *plugin)
{
	NMIodinePluginPrivate *priv = nm_iodine_plugin_get_instance_private (NM_IODINE_PLUGIN(plugin));

	priv->failure = -1;
}

static void
nm_iodine_plugin_class_init (NMIodinePluginClass *iodine_class)
{
	NMVpnServicePluginClass *parent_class = NM_VPN_SERVICE_PLUGIN_CLASS (iodine_class);

	/* virtual methods */
	parent_class->connect    = real_connect;
	parent_class->need_secrets = real_need_secrets;
	parent_class->disconnect = real_disconnect;
}

NMIodinePlugin *
nm_iodine_plugin_new (void)
{
	NMIodinePlugin *plugin;
	GError *error = NULL;

	plugin = (NMIodinePlugin *) g_initable_new (NM_TYPE_IODINE_PLUGIN, NULL, &error,
	                                            NM_VPN_SERVICE_PLUGIN_DBUS_SERVICE_NAME,
	                                            NM_DBUS_SERVICE_IODINE,
	                                            NULL);

	if (!plugin) {
		g_warning ("Failed to initialize a plugin instance: %s", error->message);
		g_error_free (error);
	}

	return plugin;
}

static void
quit_mainloop (NMIodinePlugin *plugin, gpointer user_data)
{
	g_main_loop_quit ((GMainLoop *) user_data);
}

int main (int argc, char *argv[])
{
	NMIodinePlugin *plugin;
	GMainLoop *main_loop;

#if !GLIB_CHECK_VERSION(2,36,0)
	g_type_init ();
#endif

	plugin = nm_iodine_plugin_new ();
	if (!plugin)
		exit (EXIT_FAILURE);

	main_loop = g_main_loop_new (NULL, FALSE);

	g_signal_connect (plugin, "quit",
	                  G_CALLBACK (quit_mainloop),
	                  main_loop);

	g_main_loop_run (main_loop);

	g_main_loop_unref (main_loop);
	g_object_unref (plugin);

	exit (EXIT_SUCCESS);
}
