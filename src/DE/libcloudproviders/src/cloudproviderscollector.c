/* cloudproviders.c
 *
 * Copyright (C) 2015 Carlos Soriano <csoriano@gnome.org>
 * Copyright (C) 2017 Julius Haertl <jus@bitgrid.net>
 *
 * This file is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 3 of the
 * License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cloudproviderscollector.h"
#include "cloudprovidersaccount.h"
#include "cloudprovidersprovider.h"
#include "cloudproviders-generated.h"
#include <glib.h>
#include <glib/gprintf.h>
#include <gio/gio.h>

#define KEY_FILE_GROUP "Cloud Providers"
#define XDG_DESKTOP_IMPLEMENTS "org.freedesktop.CloudProviders"

struct _CloudProvidersCollector
{
    GObject parent;

    GList *providers;
    GHashTable *providers_bus_names;
    GDBusConnection *bus;
    GCancellable *cancellable;
    GList *monitors;
};

G_DEFINE_TYPE (CloudProvidersCollector, cloud_providers_collector, G_TYPE_OBJECT)

/**
 * SECTION:cloudproviderscollector
 * @title: CloudProvidersCollector
 * @short_description: Singleton for tracking all providers.
 * @include: src/cloudproviders.h
 *
 * #CloudProvidersCollector is a singleton to track all the changes in all providers.
 * Using a #CloudProvidersCollector you can implement integration for all of them at once
 * and represent them in the UI, track new providers added or removed and their
 * status.
 */

enum
{
  PROVIDERS_CHANGED,
  LAST_SIGNAL
};

static guint signals [LAST_SIGNAL];

static void
update_cloud_providers (CloudProvidersCollector *self);

static void
on_bus_acquired (GObject      *source_object,
                 GAsyncResult *res,
                 gpointer      user_data)
{
    CloudProvidersCollector *self;
    g_autoptr(GError) error = NULL;
    GDBusConnection *connection;

    connection = g_bus_get_finish (res, &error);
    if (error != NULL)
    {
        if (!g_error_matches (error, G_IO_ERROR, G_IO_ERROR_CANCELLED))
            g_debug ("Error acquiring bus for cloud providers: %s", error->message);
        return;
    }

    /* The CloudProvidersCollector could be destroyed before arriving here */
    self = CLOUD_PROVIDERS_COLLECTOR (user_data);
    self->bus = g_steal_pointer (&connection);
    update_cloud_providers (self);
}

static void
cloud_providers_collector_dispose (GObject *object)
{
    CloudProvidersCollector *self = (CloudProvidersCollector *)object;
    GList *l;

    g_cancellable_cancel (self->cancellable);
    g_clear_object (&self->cancellable);
    for (l = self->monitors; l != NULL; l = l->next)
    {
        g_signal_handlers_disconnect_by_data (G_OBJECT (l->data), self);
    }
    for (l = self->providers; l != NULL; l = l->next)
    {
        g_signal_handlers_disconnect_by_data (G_OBJECT (l->data), self);
    }
    g_clear_pointer (&self->providers_bus_names, g_hash_table_unref);
    g_list_free_full (self->providers, g_object_unref);
    self->providers = NULL;
    g_list_free_full (self->monitors, g_object_unref);
    self->monitors = NULL;
    g_clear_object (&self->bus);

    G_OBJECT_CLASS (cloud_providers_collector_parent_class)->dispose (object);
}

static void
cloud_providers_collector_class_init (CloudProvidersCollectorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = cloud_providers_collector_dispose;

  /**
   * CloudProviderCollector::providers-changed
   *
   * This signal is emitted by the amount of providers changed.
   */
  signals [PROVIDERS_CHANGED] = g_signal_new ("providers-changed",
                                              G_TYPE_FROM_CLASS (klass),
                                              G_SIGNAL_RUN_LAST,
                                              0,
                                              NULL,
                                              NULL,
                                              g_cclosure_marshal_generic,
                                              G_TYPE_NONE,
                                              0);
}

static void
cloud_providers_collector_init (CloudProvidersCollector *self)
{
    self->cancellable = g_cancellable_new ();
    self->providers_bus_names = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);

    g_bus_get (G_BUS_TYPE_SESSION,
               self->cancellable,
               on_bus_acquired,
               self);
}

/**
 * cloud_providers_collector_get_providers
 * @self: A CloudProvidersCollector
 * Returns: (element-type CloudProviders.Provider) (transfer none): A GList* of #CloudProvidersProvider objects.
 */
GList*
cloud_providers_collector_get_providers (CloudProvidersCollector *self)
{
  return self->providers;
}

static void
on_provider_removed (CloudProvidersCollector *self)
{
    update_cloud_providers (self);
}

static void
add_cloud_provider (CloudProvidersCollector *self,
                    const gchar             *bus_name,
                    const gchar             *object_path)
{
    g_autoptr(CloudProvidersProvider) provider = NULL;

    provider = cloud_providers_provider_new (bus_name, object_path);
    if (g_hash_table_contains (self->providers_bus_names, bus_name))
    {
        g_debug ("Skipped duplicate provider: %s %s\n", bus_name, object_path);
        return;
    }

    self->providers = g_list_append (self->providers, g_object_ref (provider));
    g_hash_table_add (self->providers_bus_names, g_strdup (bus_name));
    g_signal_connect_swapped (provider, "removed",
                              G_CALLBACK (on_provider_removed), self);

    g_debug ("Client loading provider: %s %s\n", bus_name, object_path);
}

static void
load_cloud_provider_from_desktop_file (CloudProvidersCollector *self,
                                       GFile                   *file)
{
    g_autoptr(GKeyFile) key_file = NULL;
    g_autoptr(GError) error = NULL;
    g_autofree gchar *path = NULL;
    g_auto(GStrv) implements = NULL;
    g_autofree gchar *bus_name = NULL;
    g_autofree gchar *object_path = NULL;

    path = g_file_get_path (file);
    if (!g_str_has_suffix (path, ".desktop"))
    {
        return;
    }

    key_file = g_key_file_new ();
    if (!g_key_file_load_from_file (key_file, path, G_KEY_FILE_NONE, &error))
    {
        g_warning ("Error loading .desktop file at %s: %s", path, error->message);
        return;
    }

    implements = g_key_file_get_string_list (key_file, G_KEY_FILE_DESKTOP_GROUP, "Implements", NULL, &error);
    if (implements == NULL)
    {
        if (!g_error_matches (error, G_KEY_FILE_ERROR, G_KEY_FILE_ERROR_KEY_NOT_FOUND))
        {
            g_warning ("Error loading Implements key from %s: %s", path, error->message);
        }

        return;
    }

    if (!g_strv_contains((const gchar * const *) implements, XDG_DESKTOP_IMPLEMENTS))
    {
        return;
    }

    bus_name = g_key_file_get_string (key_file, XDG_DESKTOP_IMPLEMENTS, "BusName", NULL);
    object_path = g_key_file_get_string (key_file, XDG_DESKTOP_IMPLEMENTS, "ObjectPath", NULL);

    if (!bus_name || !object_path)
    {
        g_warning ("Unable to find BusName or ObjectPath keys in "XDG_DESKTOP_IMPLEMENTS" section from %s", path);
        return;
    }

    add_cloud_provider (self, bus_name, object_path);
}

static void
load_cloud_provider_from_key_file (CloudProvidersCollector *self,
                                   GFile                   *file)
{
    g_autoptr(GKeyFile) key_file = g_key_file_new ();
    g_autofree gchar *path = NULL;
    GError *error = NULL;
    g_autofree gchar *bus_name = NULL;
    g_autofree gchar *object_path = NULL;
    CloudProvidersProvider *provider;

    path = g_file_get_path (file);
    g_key_file_load_from_file (key_file, path, G_KEY_FILE_NONE, &error);
    if (error != NULL)
    {
        g_debug ("Error while loading cloud provider key file at %s with error %s", path, error->message);
        return;
    }

    if (!g_key_file_has_group (key_file, KEY_FILE_GROUP))
    {
        g_debug ("Error while loading cloud provider key file at %s with error %s", path, error->message);
        return;
    }

    bus_name = g_key_file_get_string (key_file, KEY_FILE_GROUP, "BusName", &error);
    if (error != NULL)
    {
        g_debug ("Error while loading cloud provider key file at %s with error %s", path, error->message);
        return;
    }
    object_path = g_key_file_get_string (key_file, KEY_FILE_GROUP, "ObjectPath", &error);
    if (error != NULL)
    {
        g_debug ("Error while loading cloud provider key file at %s with error %s", path, error->message);
        return;
    }

    provider = cloud_providers_provider_new (bus_name, object_path);
    self->providers = g_list_append (self->providers, provider);
    g_signal_connect_swapped (provider, "removed",
                              G_CALLBACK (on_provider_removed), self);

    g_debug("Client loading provider: %s %s\n", bus_name, object_path);
}

static void
on_providers_file_changed (CloudProvidersCollector *self)
{
    update_cloud_providers (self);
}

static void
load_cloud_providers (CloudProvidersCollector *self)
{
    const gchar* const *data_dirs;
    const gchar *data_subdirs[] = {"applications", "cloud-providers"};
    guint i;
    gsize j;

    data_dirs = g_get_system_data_dirs ();
    for (i = 0; data_dirs[i] != NULL; i++)
    {
        for (j = 0; j < G_N_ELEMENTS (data_subdirs); j++)
        {
            g_autoptr (GFile) directory_file = NULL;
            g_autoptr (GFileInfo) info = NULL;
            g_autoptr (GFileEnumerator) file_enumerator = NULL;
            g_autoptr (GFileMonitor) monitor = NULL;
            g_autoptr (GError) error = NULL;

            directory_file = g_file_new_build_filename (data_dirs[i], data_subdirs[j], NULL);
            monitor = g_file_monitor (directory_file, G_FILE_MONITOR_WATCH_MOVES,
                                      self->cancellable, NULL);
            g_signal_connect_swapped (monitor, "changed", G_CALLBACK (on_providers_file_changed), self);
            self->monitors = g_list_append (self->monitors, g_steal_pointer (&monitor));
            file_enumerator = g_file_enumerate_children (directory_file,
                                                         "standard::name,standard::type",
                                                         G_FILE_QUERY_INFO_NONE,
                                                         NULL,
                                                         &error);
            if (error)
            {
                g_clear_error (&error);
                continue;
            }

            info = g_file_enumerator_next_file (file_enumerator, NULL, &error);
            if (error)
            {
                g_autofree gchar *directory_path = g_file_get_path (directory_file);
                g_warning ("Error while enumerating file %s error: %s\n", directory_path, error->message);
                g_clear_error (&error);
                continue;
            }

            while (info != NULL && error == NULL)
            {
                g_autoptr(GFile) child = g_file_enumerator_get_child (file_enumerator, info);
                if (strcmp (data_subdirs[j], "applications") == 0)
                {
                    load_cloud_provider_from_desktop_file (self, child);
                }
                else
                {
                    load_cloud_provider_from_key_file (self, child);
                }

                g_clear_object (&info);
                info = g_file_enumerator_next_file (file_enumerator, NULL, &error);
            }
        }
    }
}

static void
update_cloud_providers (CloudProvidersCollector *self)
{
    GList *l;

    g_cancellable_cancel (self->cancellable);
    g_clear_object (&self->cancellable);
    self->cancellable = g_cancellable_new ();
    for (l = self->monitors; l != NULL; l = l->next)
    {
        g_signal_handlers_disconnect_by_data (G_OBJECT (l->data), self);
    }
    for (l = self->providers; l != NULL; l = l->next)
    {
        g_signal_handlers_disconnect_by_data (G_OBJECT (l->data), self);
    }
    g_hash_table_remove_all (self->providers_bus_names);
    g_list_free_full (self->providers, g_object_unref);
    g_list_free_full (self->monitors, g_object_unref);
    self->providers = NULL;
    self->monitors = NULL;

    load_cloud_providers (self);

    g_signal_emit_by_name (G_OBJECT (self), "providers-changed");
}

static gpointer
singleton_creation_thread (gpointer data)
{
    return g_object_new (CLOUD_PROVIDERS_TYPE_COLLECTOR, NULL);
}

/**
 * cloud_providers_collector_dup_singleton:
 * Main object to track changes in all providers.
 *
 * Returns: (transfer full): A manager singleton
 */
CloudProvidersCollector *
cloud_providers_collector_dup_singleton (void)
{
    static GOnce collector_singleton = G_ONCE_INIT;
    CloudProvidersCollector *self;

    g_once (&collector_singleton, singleton_creation_thread, NULL);

    self = CLOUD_PROVIDERS_COLLECTOR (collector_singleton.retval);
    return g_object_ref (self);
}

