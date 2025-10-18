/* cloudprovidersprovider.c
 *
 * Copyright (C) 2017 Carlos Soriano <csoriano@gnome.org>
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

#include <gio/gio.h>

#include "cloudprovidersprovider.h"
#include "cloudprovidersaccount.h"
#include "cloudproviders-generated.h"

struct _CloudProvidersProvider
{
    GObject parent_instance;

    GDBusObjectManager *manager;
    gchar *name;
    GList *accounts;
    gchar *manager_bus_name;
    gchar *manager_object_path;

    GDBusConnection *bus;
    CloudProvidersDbusProvider *proxy;
    GCancellable *cancellable;
};

G_DEFINE_TYPE (CloudProvidersProvider, cloud_providers_provider, G_TYPE_OBJECT)

/**
 * SECTION:cloudprovidersprovider
 * @title: CloudProvidersProvider
 * @short_description: Base object for representing a single provider for clients.
 * @include: src/cloudprovidersprovider.h
 *
 * #CloudProvidersProvider is the basic object object for client implementers
 * that contains the properties of the provider such as name, and the accounts
 * associated with it.
 */

enum
{
    PROP_0,
    PROP_NAME,
    N_PROPS
};

enum
{
    ACCOUNTS_CHANGED,
    REMOVED,
    LAST_SIGNAL
};

static guint signals [LAST_SIGNAL];
static GParamSpec *properties [N_PROPS];

static void
on_name_changed (CloudProvidersProvider *self);
static void
on_cloud_providers_object_manager_object_added (GDBusObjectManager *manager,
                                                GDBusObject        *object,
                                                gpointer            user_data);

static void
on_cloud_providers_object_manager_object_removed (GDBusObjectManager *manager,
                                                  GDBusObject        *object,
                                                  gpointer            user_data);

static void
on_cloud_providers_object_manager_name_owner_changed (GObject    *object,
                                                      GParamSpec *pspec,
                                                      gpointer    user_data);

CloudProvidersProvider *
cloud_providers_provider_new (const gchar *bus_name,
                              const gchar *object_path)
{
    CloudProvidersProvider *self;

    g_return_val_if_fail (bus_name != NULL, NULL);
    g_return_val_if_fail (object_path != NULL, NULL);

    self = g_object_new (CLOUD_PROVIDERS_TYPE_PROVIDER, NULL);
    self->manager_bus_name = g_strdup (bus_name);
    self->manager_object_path = g_strdup (object_path);

    return self;
}

static void
cloud_providers_provider_dispose (GObject *object)
{
    CloudProvidersProvider *self = (CloudProvidersProvider *)object;

    g_cancellable_cancel (self->cancellable);
    g_clear_object (&self->cancellable);

    g_list_free_full (self->accounts, g_object_unref);
    self->accounts = NULL;

    if (self->proxy)
        g_signal_handlers_disconnect_by_func (self->proxy, G_CALLBACK (on_name_changed), self);
    g_clear_object (&self->proxy);

    if (self->manager)
    {
        g_signal_handlers_disconnect_by_func (self->manager,
                                              G_CALLBACK (on_cloud_providers_object_manager_name_owner_changed),
                                              self);
        g_signal_handlers_disconnect_by_func (self->manager,
                                              G_CALLBACK (on_cloud_providers_object_manager_object_added),
                                              self);
        g_signal_handlers_disconnect_by_func (self->manager,
                                              G_CALLBACK (on_cloud_providers_object_manager_object_removed),
                                              self);
    }

    g_clear_object (&self->manager);
    g_clear_object (&self->bus);

    G_OBJECT_CLASS (cloud_providers_provider_parent_class)->dispose (object);
}

static void
cloud_providers_provider_finalize (GObject *object)
{
    CloudProvidersProvider *self = (CloudProvidersProvider *)object;

    g_clear_pointer (&self->name, g_free);
    g_clear_pointer (&self->manager_bus_name, g_free);
    g_clear_pointer (&self->manager_object_path, g_free);

    G_OBJECT_CLASS (cloud_providers_provider_parent_class)->finalize (object);
}

static void
cloud_providers_provider_get_property (GObject    *object,
                                       guint       prop_id,
                                       GValue     *value,
                                       GParamSpec *pspec)
{
    CloudProvidersProvider *self = CLOUD_PROVIDERS_PROVIDER (object);

    switch (prop_id)
    {
        case PROP_NAME:
        {
            g_value_set_string (value, self->name);
        }
        break;

        default:
        {
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        }

    }
}

static void
cloud_providers_provider_set_property (GObject      *object,
                                       guint         prop_id,
                                       const GValue *value,
                                       GParamSpec   *pspec)
{
    CloudProvidersProvider *self = CLOUD_PROVIDERS_PROVIDER (object);

    switch (prop_id)
    {
        case PROP_NAME:
        {
            g_free (self->name);
            self->name = g_value_dup_string (value);
            g_object_notify_by_pspec (object, properties[PROP_NAME]);
        }
        break;

        break;
        default:
        {
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        }

    }
}

static void
cloud_providers_provider_class_init (CloudProvidersProviderClass *klass)
{
    GObjectClass *object_class = G_OBJECT_CLASS (klass);

    object_class->dispose = cloud_providers_provider_dispose;
    object_class->finalize = cloud_providers_provider_finalize;
    object_class->get_property = cloud_providers_provider_get_property;
    object_class->set_property = cloud_providers_provider_set_property;

    properties [PROP_NAME] =
        g_param_spec_string ("name",
                             "Name",
                             "Name of the provider",
                             NULL,
                             (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
    g_object_class_install_properties (object_class,
                                       N_PROPS,
                                       properties);

  /**
   * CloudProviderProvider::accounts-changed
   *
   * This signal is emitted by a provider if the number of accounts changed.
   */
  signals [ACCOUNTS_CHANGED] =
    g_signal_new ("accounts-changed",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL,
                  NULL,
                  g_cclosure_marshal_generic,
                  G_TYPE_NONE,
                  0);
  /**
   * CloudProviderProvider::removed
   *
   * This signal is emitted by a provider when the provider is removed in DBUS.
   */
  signals [REMOVED] =
    g_signal_new ("removed",
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
update_cloud_providers_accounts (CloudProvidersProvider *self)
{
    GList *objects;
    GList *l;

    g_list_free_full (self->accounts, g_object_unref);
    self->accounts = NULL;
    objects = g_dbus_object_manager_get_objects (self->manager);
    if (objects == NULL)
    {
        if (self->name != NULL)
        {
            g_debug ("Provider accounts not ready server side for %s\n", self->name);
        }
        else
        {
            g_debug ("Provider accounts not ready server side for %p\n", self);
        }
        return;
    }

    for (l = objects; l != NULL; l = l->next)
    {
        CloudProvidersDbusObject *object;
        GDBusInterface *interface;

        object = CLOUD_PROVIDERS_DBUS_OBJECT (l->data);
        interface = g_dbus_object_get_interface (G_DBUS_OBJECT (object),
                                                 CLOUD_PROVIDERS_PROVIDER_DBUS_IFACE);
        if (interface != NULL)
        {
            self->proxy = cloud_providers_dbus_object_get_provider (object);
            g_signal_connect_swapped (self->proxy, "notify::name", G_CALLBACK (on_name_changed), self);
            on_name_changed (self);
            g_object_unref (interface);

            continue;
        }
        interface = g_dbus_object_get_interface (G_DBUS_OBJECT (object),
                                                 CLOUD_PROVIDERS_ACCOUNT_DBUS_IFACE);
        if (interface != NULL)
        {
            CloudProvidersAccount *account;

            account = cloud_providers_account_new (G_DBUS_PROXY (cloud_providers_dbus_object_peek_account (object)));

            self->accounts = g_list_append (self->accounts, account);
            g_object_unref (interface);

            continue;
        }
    }

    g_list_free_full (objects, g_object_unref);

    g_signal_emit_by_name (G_OBJECT (self), "accounts-changed", NULL);
}

static void
on_cloud_providers_object_manager_object_added (GDBusObjectManager *manager,
                                                GDBusObject        *object,
                                                gpointer            user_data)
{
    CloudProvidersProvider *self;

    self = CLOUD_PROVIDERS_PROVIDER (user_data);
    update_cloud_providers_accounts (self);
}

static void
on_cloud_providers_object_manager_object_removed (GDBusObjectManager *manager,
                                                  GDBusObject        *object,
                                                  gpointer            user_data)
{
    CloudProvidersProvider *self;

    self = CLOUD_PROVIDERS_PROVIDER (user_data);
    update_cloud_providers_accounts (self);
}

static void
on_cloud_providers_object_manager_name_owner_changed (GObject    *object,
                                                      GParamSpec *pspec,
                                                      gpointer    user_data)
{
    CloudProvidersProvider *self;

    self = CLOUD_PROVIDERS_PROVIDER (user_data);
    g_signal_emit_by_name (G_OBJECT (self), "removed", NULL);
}

static void
on_name_changed (CloudProvidersProvider *self)
{
    g_free (self->name);
    self->name = cloud_providers_dbus_provider_dup_name (self->proxy);
    if (self->name == NULL)
    {
        g_debug ("Provider name not ready for %p\n", self);
        return;
    }

    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_NAME]);
}

static void
on_object_manager_created (GObject      *source_object,
                           GAsyncResult *res,
                           gpointer      user_data)
{
    CloudProvidersProvider *self;
    g_autoptr(GError) error = NULL;
    GDBusObjectManager *manager;

    manager = cloud_providers_dbus_object_manager_client_new_finish (res, &error);
    if (error != NULL)
    {
        g_printerr ("Error getting object manager client: %s", error->message);
        return;
    }

    /* The CloudProvidersProvider could be destroyed before arriving here */
    self = CLOUD_PROVIDERS_PROVIDER (user_data);
    self->manager = g_steal_pointer (&manager);
    g_signal_connect (self->manager, "notify::name-owner",
                      G_CALLBACK (on_cloud_providers_object_manager_name_owner_changed),
                      self);
    g_signal_connect (self->manager, "object-added",
                      G_CALLBACK (on_cloud_providers_object_manager_object_added),
                      self);
    g_signal_connect (self->manager, "object-removed",
                      G_CALLBACK (on_cloud_providers_object_manager_object_removed),
                      self);

    update_cloud_providers_accounts (self);
}

static void
on_bus_acquired (GObject      *source_object,
                 GAsyncResult *res,
                 gpointer      user_data)
{
    CloudProvidersProvider *self;
    g_autoptr(GError) error = NULL;
    GDBusConnection *connection;

    connection = g_bus_get_finish (res, &error);
    if (error != NULL)
    {
        if (!g_error_matches (error, G_IO_ERROR, G_IO_ERROR_CANCELLED))
            g_debug ("Error acquiring bus for cloud provider: %s", error->message);
        return;
    }

    /* The CloudProvidersProvider could be destroyed before arriving here */
    self = CLOUD_PROVIDERS_PROVIDER (user_data);
    self->bus = g_steal_pointer (&connection);
    cloud_providers_dbus_object_manager_client_new (self->bus,
                                                    G_DBUS_OBJECT_MANAGER_CLIENT_FLAGS_NONE,
                                                    self->manager_bus_name,
                                                    self->manager_object_path,
                                                    self->cancellable,
                                                    on_object_manager_created,
                                                    self);
}


static void
cloud_providers_provider_init (CloudProvidersProvider *self)
{
    self->cancellable = g_cancellable_new ();
    g_bus_get (G_BUS_TYPE_SESSION,
               self->cancellable,
               on_bus_acquired,
               self);
}


const gchar*
cloud_providers_provider_get_name (CloudProvidersProvider *self)
{
    g_return_val_if_fail (CLOUD_PROVIDERS_IS_PROVIDER (self), NULL);

    return self->name;
}

/**
 * cloud_providers_provider_get_accounts
 * @self: A CloudProvidersProvider
 * Returns: (element-type CloudProviders.Account) (transfer none): A GList* of #CloudProvidersProvider objects.
 */
GList*
cloud_providers_provider_get_accounts (CloudProvidersProvider *self)
{
    g_return_val_if_fail (CLOUD_PROVIDERS_IS_PROVIDER (self), NULL);

    return self->accounts;
}


