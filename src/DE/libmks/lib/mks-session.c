/*
 * mks-session.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "mks-device-private.h"
#include "mks-read-only-list-model-private.h"
#include "mks-qemu.h"
#include "mks-screen.h"
#include "mks-session.h"

/**
 * MksSession:
 *
 * Session connected to a QEMU VM
 *
 * The `MksSession` represents a connection to a QEMU VM instance. It contains
 * devices such as the mouse, keyboard, and screen which can be used with GTK.
 *
 * You may monitor [property@Mks.Session:devices] using [signal@Gio.ListModel::items-changed] to be
 * notified of changes to available devices in the session.
 *
 * # Connecting To QEMU
 *
 * To use `MksSession`, you should create your QEMU instance using `dbus` for
 * the various devices that support it. You'll need to provide your P2P D-Bus
 * address when connecting to QEMU.
 *
 * Using the same [class@Gio.DBusConnection], create a `MksSession` with
 * [func@Mks.Session.new_for_connection]. The `MksSession` instance will negotiate
 * with the peer to determine what devices are available and expose them
 * via the [property@Mks.Session:devices] [iface@Gio.ListModel].
 *
 * # Creating Widgets
 *
 * You can create a new widget to embed in your application by calling
 * [method@Mks.Session.ref_screen] and set the screen for the [class@Mks.Display]
 * with [method@Mks.Display.set_screen].
 */

#define QEMU_BUS_NAME "org.qemu"

static gboolean mks_session_initable_init              (GInitable            *initable,
                                                        GCancellable         *cancellable,
                                                        GError              **error);
static void     mks_session_async_initable_init_async  (GAsyncInitable       *async_initable,
                                                        int                   io_priority,
                                                        GCancellable         *cancellable,
                                                        GAsyncReadyCallback   callback,
                                                        gpointer              user_data);
static gboolean mks_session_async_initable_init_finish (GAsyncInitable       *async_initable,
                                                        GAsyncResult         *result,
                                                        GError              **error);

struct _MksSession
{
  GObject parent_instance;

  /* @connection is used to communicate with the QEMU instance. It is expected
   * to be a private point-to-point connection over a Unix socket, socketpair(),
   * or other channel capable of passing FDs between peers.
   */
  GDBusConnection *connection;

  /* As devices are discovered from the QEMU instance, MksDevice-based objects
   * are created for them and stored in @devices. Applications will work with
   * these objects to perform operations on the QEMU instance. When we discover
   * the devices have been removed, we drop them from the listmodel.
   */
  GListStore *devices;

  /* @devices_read_only is a #MksReadOnlyListModel of @devices so that we may
   * allow observation of the #GListModel, but without access to perturb it.
   * Such opaqueness provides us a bit more assurance we don't leak
   * implementation details into the API.
   */
  GListModel *devices_read_only;

  /* An object manager client is used to monitor for new objects exported by
   * the QEMU instance. Those objects are then wrapped by MksDevice objects
   * as necessary and exported to consumers via @devices.
   */
  GDBusObjectManager *object_manager;

  /* A GDBusObjectProxy to the `/org/qemu/Display1/VM` instance. Generally
   * this used via the `org.qemu.Display1.VM` interface. We track the
   * top-level object-manager instance so that we can detect easily when
   * the VM has been removed from the peer. (Which could happen if it is
   * running on a private D-Bus rather than a socketpair().
   */
  MksQemuObject *vm_object;
  MksQemuVM *vm;

  char  *name;
  char  *uuid;
  char  *bus_name;
};

static void
initable_iface_init (GInitableIface *iface)
{
  iface->init = mks_session_initable_init;
}

static void
async_initable_iface_init (GAsyncInitableIface *iface)
{
  iface->init_async = mks_session_async_initable_init_async;
  iface->init_finish = mks_session_async_initable_init_finish;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (MksSession, mks_session, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE, initable_iface_init)
                               G_IMPLEMENT_INTERFACE (G_TYPE_ASYNC_INITABLE, async_initable_iface_init))

enum {
  PROP_0,
  PROP_CONNECTION,
  PROP_BUS_NAME,
  PROP_DEVICES,
  PROP_NAME,
  PROP_UUID,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

static void
mks_session_set_connection (MksSession      *self,
                            GDBusConnection *connection)
{
  g_assert (MKS_IS_SESSION (self));
  g_assert (!connection || G_IS_DBUS_CONNECTION (connection));
  g_assert (self->connection == NULL);

  if (connection == NULL)
    g_critical ("%s created without a GDBusConnection, this cannot work.",
                G_OBJECT_TYPE_NAME (self));
  else
    self->connection = g_object_ref (connection);
}

static void
mks_session_add_device (MksSession *self,
                        MksDevice  *device)
{
  g_assert (MKS_IS_SESSION (self));
  g_assert (!device || MKS_IS_DEVICE (device));

  if (device == NULL)
    return;

  g_list_store_append (self->devices, device);
}

static void
mks_session_set_name (MksSession *self,
                      const char *name)
{
  g_assert (MKS_IS_SESSION (self));

  if (g_set_str (&self->name, name))
    g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_NAME]);
}

static void
mks_session_set_uuid (MksSession *self,
                      const char *uuid)
{
  g_assert (MKS_IS_SESSION (self));

  if (g_set_str (&self->uuid, uuid))
    g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_UUID]);
}

static void
mks_session_vm_notify_cb (MksSession *self,
                          GParamSpec *pspec,
                          MksQemuVM  *vm)
{
  g_assert (MKS_IS_SESSION (self));
  g_assert (pspec != NULL);
  g_assert (MKS_QEMU_IS_VM (vm));

  if (strcmp (pspec->name, "name") == 0)
    mks_session_set_name (self, mks_qemu_vm_get_name (vm));
  else if (strcmp (pspec->name, "uuid") == 0)
    mks_session_set_uuid (self, mks_qemu_vm_get_uuid (vm));
}

static void
mks_session_set_vm (MksSession    *self,
                    MksQemuObject *vm_object,
                    MksQemuVM     *vm)
{
  g_assert (MKS_IS_SESSION (self));
  g_assert (MKS_QEMU_IS_OBJECT (vm_object));
  g_assert (MKS_QEMU_IS_VM (vm));

  if (self->vm != NULL)
    return;

  g_set_object (&self->vm_object, vm_object);
  g_set_object (&self->vm, vm);

  g_signal_connect_object (vm,
                           "notify",
                           G_CALLBACK (mks_session_vm_notify_cb),
                           self,
                           G_CONNECT_SWAPPED);
  mks_session_set_name (self, mks_qemu_vm_get_name (vm));
  mks_session_set_uuid (self, mks_qemu_vm_get_uuid (vm));
}

static void
mks_session_object_manager_object_added_cb (MksSession         *self,
                                            MksQemuObject      *object,
                                            GDBusObjectManager *manager)
{
  g_autolist(GDBusInterface) interfaces = NULL;

  g_assert (MKS_IS_SESSION (self));
  g_assert (MKS_QEMU_IS_OBJECT (object));
  g_assert (G_IS_DBUS_OBJECT_MANAGER (manager));

  interfaces = g_dbus_object_get_interfaces (G_DBUS_OBJECT (object));

  for (const GList *iter = interfaces; iter; iter = iter->next)
    {
      GDBusInterface *iface = iter->data;

      if (MKS_QEMU_IS_VM (iface))
        mks_session_set_vm (self, object, MKS_QEMU_VM (iface));
      else if (MKS_QEMU_IS_CONSOLE (iface))
        mks_session_add_device (self, _mks_device_new (MKS_TYPE_SCREEN, self, object));
    }
}

static void
mks_session_object_manager_object_removed_cb (MksSession         *self,
                                              MksQemuObject      *object,
                                              GDBusObjectManager *manager)
{
  g_assert (MKS_IS_SESSION (self));
  g_assert (MKS_QEMU_IS_OBJECT (object));
  g_assert (G_IS_DBUS_OBJECT_MANAGER (manager));

  if (object == self->vm_object)
    {
      g_clear_object (&self->vm);
      g_clear_object (&self->vm_object);
      g_list_store_remove_all (self->devices);
    }
}

static void
mks_session_set_object_manager (MksSession         *self,
                                GDBusObjectManager *object_manager)
{
  g_assert (MKS_IS_SESSION (self));
  g_assert (!object_manager || MKS_QEMU_IS_OBJECT_MANAGER_CLIENT (object_manager));
  g_assert (self->object_manager == NULL);

  if (g_set_object (&self->object_manager, object_manager))
    {
      g_autolist(GDBusObjectProxy) objects = NULL;

      g_signal_connect_object (object_manager,
                               "object-added",
                               G_CALLBACK (mks_session_object_manager_object_added_cb),
                               self,
                               G_CONNECT_SWAPPED);
      g_signal_connect_object (object_manager,
                               "object-removed",
                               G_CALLBACK (mks_session_object_manager_object_removed_cb),
                               self,
                               G_CONNECT_SWAPPED);

      objects = g_dbus_object_manager_get_objects (object_manager);
      for (const GList *iter = objects; iter; iter = iter->next)
        mks_session_object_manager_object_added_cb (self, iter->data, object_manager);
    }
}

static void
mks_session_dispose (GObject *object)
{
  MksSession *self = (MksSession *)object;

  if (self->devices != NULL)
    g_list_store_remove_all (self->devices);

  g_clear_object (&self->object_manager);
  g_clear_object (&self->vm);
  g_clear_object (&self->vm_object);
  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->uuid, g_free);
  g_clear_pointer (&self->bus_name, g_free);

  G_OBJECT_CLASS (mks_session_parent_class)->dispose (object);
}

static void
mks_session_finalize (GObject *object)
{
  MksSession *self = (MksSession *)object;

  g_clear_object (&self->devices);
  g_clear_object (&self->devices_read_only);
  g_clear_object (&self->connection);

  G_OBJECT_CLASS (mks_session_parent_class)->finalize (object);
}

static void
mks_session_get_property (GObject    *object,
                          guint       prop_id,
                          GValue     *value,
                          GParamSpec *pspec)
{
  MksSession *self = MKS_SESSION (object);

  switch (prop_id)
    {
    case PROP_CONNECTION:
      g_value_set_object (value, mks_session_get_connection (self));
      break;

    case PROP_BUS_NAME:
      g_value_set_string (value, self->bus_name);
      break;

    case PROP_DEVICES:
      g_value_set_object (value, mks_session_get_devices (self));
      break;

    case PROP_NAME:
      g_value_set_string (value, mks_session_get_name (self));
      break;

    case PROP_UUID:
      g_value_set_string (value, mks_session_get_uuid (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_session_set_property (GObject      *object,
                          guint         prop_id,
                          const GValue *value,
                          GParamSpec   *pspec)
{
  MksSession *self = MKS_SESSION (object);

  switch (prop_id)
    {
    case PROP_CONNECTION:
      mks_session_set_connection (self, g_value_get_object (value));
      break;
    case PROP_BUS_NAME:
      g_set_str (&self->bus_name, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_session_class_init (MksSessionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = mks_session_dispose;
  object_class->finalize = mks_session_finalize;
  object_class->get_property = mks_session_get_property;
  object_class->set_property = mks_session_set_property;

  /**
   * MksSession:connection:
   *
   * The [class@Gio.DBusConnection] that is used to communicate with QEMU.
   */
  properties [PROP_CONNECTION] =
    g_param_spec_object ("connection", NULL, NULL,
                         G_TYPE_DBUS_CONNECTION,
                         (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  /**
   * MksSession:bus-name:
   *
   * The unique connection name to connect to or `org.qemu` as fallback.
   */
  properties [PROP_BUS_NAME] =
    g_param_spec_string ("bus-name", NULL, NULL,
                         QEMU_BUS_NAME,
                         (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  /**
   * MksSession:devices:
   *
   * A [iface@Gio.ListModel] of devices that have been
   * discovered on the [class@Gio.DBusConnection] to QEMU.
   */
  properties [PROP_DEVICES] =
    g_param_spec_object ("devices", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  /**
   * MksSession:name:
   *
   * The VM name as specified by the QEMU instance.
   */
  properties [PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  /**
   * MksSession:uuid:
   *
   * The VM unique identifier specified by the QEMU instance.
   */
  properties [PROP_UUID] =
    g_param_spec_string ("uuid", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
mks_session_init (MksSession *self)
{
  self->devices = g_list_store_new (MKS_TYPE_DEVICE);
  self->devices_read_only = mks_read_only_list_model_new (G_LIST_MODEL (self->devices));
}

static gboolean
mks_session_initable_init (GInitable     *initable,
                           GCancellable  *cancellable,
                           GError       **error)
{
  MksSession *self = (MksSession *)initable;
  g_autoptr(GDBusObjectManager) object_manager = NULL;

  g_assert (MKS_IS_SESSION (self));
  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));

  if (self->connection == NULL)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_NOT_CONNECTED,
                   "Not connected");
      return FALSE;
    }

  object_manager =
    mks_qemu_object_manager_client_new_sync (self->connection,
                                             G_DBUS_OBJECT_MANAGER_CLIENT_FLAGS_DO_NOT_AUTO_START,
                                             self->bus_name,
                                             "/org/qemu/Display1",
                                             cancellable,
                                             error);

  mks_session_set_object_manager (self, object_manager);

  return object_manager != NULL;
}

static void
mks_session_async_initable_vm_cb (GObject      *object,
                                  GAsyncResult *result,
                                  gpointer      user_data)
{
  g_autoptr(GDBusObjectManager) object_manager = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GTask) task = user_data;
  MksSession *self;

  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (G_IS_TASK (task));

  self = g_task_get_source_object (task);
  object_manager = mks_qemu_object_manager_client_new_finish (result, &error);

  g_assert (MKS_IS_SESSION (self));
  g_assert (!object_manager || MKS_QEMU_IS_OBJECT_MANAGER_CLIENT (object_manager));

  mks_session_set_object_manager (self, object_manager);

  if (error != NULL)
    g_task_return_error (task, g_steal_pointer (&error));
  else
    g_task_return_boolean (task, TRUE);
}

static void
mks_session_async_initable_init_async (GAsyncInitable      *async_initable,
                                       int                  io_priority,
                                       GCancellable        *cancellable,
                                       GAsyncReadyCallback  callback,
                                       gpointer             user_data)
{
  MksSession *self = (MksSession *)async_initable;
  g_autoptr(GTask) task = NULL;

  g_assert (MKS_IS_SESSION (self));
  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));

  task = g_task_new (self, cancellable, callback, user_data);
  g_task_set_source_tag (task, mks_session_async_initable_init_async);
  g_task_set_priority (task, io_priority);

  if (self->connection == NULL)
    g_task_return_new_error (task,
                             G_IO_ERROR,
                             G_IO_ERROR_NOT_CONNECTED,
                             "Not connected");
  else
    mks_qemu_object_manager_client_new (self->connection,
                                        G_DBUS_OBJECT_MANAGER_CLIENT_FLAGS_DO_NOT_AUTO_START,
                                        self->bus_name,
                                        "/org/qemu/Display1",
                                        cancellable,
                                        mks_session_async_initable_vm_cb,
                                        g_steal_pointer (&task));
}

static gboolean
mks_session_async_initable_init_finish (GAsyncInitable  *async_initable,
                                        GAsyncResult    *result,
                                        GError         **error)
{
  MksSession *self = (MksSession *)async_initable;

  g_assert (MKS_IS_SESSION (self));
  g_assert (G_IS_TASK (result));

  return g_task_propagate_boolean (G_TASK (result), error);
}

/**
 * mks_session_get_devices:
 * @self: a #MksSession
 *
 * Gets a #GListModel of devices connected to the session.
 *
 * Returns: (transfer none): a #GListModel of #MksDevice
 */
GListModel *
mks_session_get_devices (MksSession *self)
{
  g_return_val_if_fail (MKS_IS_SESSION (self), NULL);

  return self->devices_read_only;
}

static void
mks_session_new_for_connection_cb (GObject      *object,
                                   GAsyncResult *result,
                                   gpointer      user_data)
{
  MksSession *self = (MksSession *)object;
  g_autoptr(GError) error = NULL;
  g_autoptr(GTask) task = user_data;

  g_assert (MKS_IS_SESSION (self));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (G_IS_TASK (task));

  if (!g_async_initable_init_finish (G_ASYNC_INITABLE (self), result, &error))
    g_task_return_error (task, g_steal_pointer (&error));
  else
    g_task_return_pointer (task, g_object_ref (self), g_object_unref);
}

/**
 * mks_session_new_for_connection:
 * @connection: a #GDBusConnection
 * @io_priority: priority for IO operations
 * @cancellable: (nullable): a #GCancellable or %NULL
 * @callback: a callback to execute upon completion of the operation
 * @user_data: closure data for @callback
 *
 * Creates a #MksSession which communicates using @connection.
 *
 * The [class@Gio.DBusConnection] should be a private D-Bus connection to a QEMU
 * instance which has devices created using the "dbus" backend.

 * @callback will be executed when the session has been created or
 * failed to create.
 *
 * This function will not block the calling thread.
 *
 * use [ctor@Mks.Session.new_for_connection_finish] to get the result of
 * this operation.
 */
void
mks_session_new_for_connection (GDBusConnection     *connection,
                                int                  io_priority,
                                GCancellable        *cancellable,
                                GAsyncReadyCallback  callback,
                                gpointer             user_data)
{
  mks_session_new_for_connection_with_name (connection,
                                            QEMU_BUS_NAME,
                                            io_priority,
                                            cancellable,
                                            callback,
                                            user_data);
}

/**
 * mks_session_new_for_connection_finish:
 * @result: a #GAsyncResult provided to callback
 * @error: (nullable): a location for a #GError or %NULL
 *
 * Completes a request to create a #MksSession for a [class@Gio.DBusConnection].
 *
 * Returns: (transfer full): a #MksSession if successful; otherwise %NULL
 *   and @error is set.
 */
MksSession *
mks_session_new_for_connection_finish (GAsyncResult  *result,
                                       GError       **error)
{
  g_return_val_if_fail (G_IS_TASK (result), NULL);

  return g_task_propagate_pointer (G_TASK (result), error);
}

/**
 * mks_session_new_for_connection_sync:
 * @connection: a private #GDBusConnetion to a QEMU instance
 * @cancellable: (nullable): a #GCancellable, or %NULL
 * @error: (nullable): a location for a #GError, or %NULL
 *
 * Synchronously creates a new #MksSession instance.
 *
 * This may block while the QEMU instance is contacted to query for
 * initial devices and VM status.
 *
 * Returns: (transfer full): a #MksSession if successful; otherwise %NULL
 *   and @error is set.
 */
MksSession *
mks_session_new_for_connection_sync (GDBusConnection  *connection,
                                     GCancellable     *cancellable,
                                     GError          **error)
{
  return mks_session_new_for_connection_with_name_sync (connection,
                                                        QEMU_BUS_NAME,
                                                        cancellable,
                                                        error);
}

/**
 * mks_session_new_for_connection_with_name:
 * @connection: a #GDBusConnection
 * @bus_name: The unique name to connect
 * @io_priority: priority for IO operations
 * @cancellable: (nullable): a #GCancellable or %NULL
 * @callback: a callback to execute upon completion of the operation
 * @user_data: closure data for @callback
 *
 * Creates a #MksSession which communicates using @connection.
 *
 * The constructor is similar to [func@Mks.Session.new_for_connection] but allows
 * to set the bus name to something else than the default `org.qemu`.
 *
 * use [ctor@Mks.Session.new_for_connection_with_name_finish] to get the result of
 * this operation.
 */
void
mks_session_new_for_connection_with_name (GDBusConnection     *connection,
                                          const char          *bus_name,
                                          int                  io_priority,
                                          GCancellable        *cancellable,
                                          GAsyncReadyCallback  callback,
                                          gpointer             user_data)
{
  g_autoptr(MksSession) self = NULL;
  g_autoptr(GTask) task = NULL;

  g_return_if_fail (G_IS_DBUS_CONNECTION (connection));
  g_return_if_fail (!cancellable || G_IS_CANCELLABLE (cancellable));
  g_return_if_fail (bus_name != NULL);

  self = g_object_new (MKS_TYPE_SESSION,
                       "connection", connection,
                       "bus-name", bus_name,
                       NULL);

  task = g_task_new (self, cancellable, callback, user_data);
  g_task_set_source_tag (task, mks_session_new_for_connection);
  g_task_set_priority (task, io_priority);

  g_async_initable_init_async (G_ASYNC_INITABLE (self),
                               io_priority,
                               cancellable,
                               mks_session_new_for_connection_cb,
                               g_steal_pointer (&task));
}

/**
 * mks_session_new_for_connection_with_name_finish:
 * @result: a #GAsyncResult provided to callback
 * @error: (nullable): a location for a #GError or %NULL
 *
 * Completes a request to create a #MksSession for a [class@Gio.DBusConnection].
 *
 * Returns: (transfer full): a #MksSession if successful; otherwise %NULL
 *   and @error is set.
 */
MksSession *
mks_session_new_for_connection_with_name_finish (GAsyncResult  *result,
                                                 GError       **error)
{
  g_return_val_if_fail (G_IS_TASK (result), NULL);

  return g_task_propagate_pointer (G_TASK (result), error);
}

/**
 * mks_session_new_for_connection_with_name_sync:
 * @bus_name: The unique name to connect
 * @connection: a private #GDBusConnetion to a QEMU instance
 * @cancellable: (nullable): a #GCancellable, or %NULL
 * @error: (nullable): a location for a #GError, or %NULL
 *
 * Synchronously creates a new #MksSession instance.
 *
 * The constructor is similar to [func@Mks.Session.new_for_connection_sync] except
 * it allows to set the bus name to something else than the default `org.qemu`.
 *
 * Returns: (transfer full): a #MksSession if successful; otherwise %NULL
 *   and @error is set.
 */
MksSession *
mks_session_new_for_connection_with_name_sync (GDBusConnection  *connection,
                                               const char       *bus_name,
                                               GCancellable     *cancellable,
                                               GError          **error)
{
  g_autoptr(MksSession) self = NULL;

  g_return_val_if_fail (G_IS_DBUS_CONNECTION (connection), NULL);
  g_return_val_if_fail (bus_name != NULL, NULL);

  self = g_object_new (MKS_TYPE_SESSION,
                       "connection", connection,
                       "bus-name", bus_name,
                       NULL);

  if (g_initable_init (G_INITABLE (self), cancellable, error))
    return g_steal_pointer (&self);

  return NULL;
}

/**
 * mks_session_get_connection:
 * @self: a #MksSession
 *
 * Gets the DBus connection used for this session.
 *
 * Returns: (transfer none) (nullable): a #GDBusConnection or %NULL if
 *   the connection has not been set, or was disposed.
 */
GDBusConnection *
mks_session_get_connection (MksSession *self)
{
  g_return_val_if_fail (MKS_IS_SESSION (self), NULL);

  return self->connection;
}

/**
 * mks_session_get_bus_name:
 * @self: a #MksSession
 *
 * Gets the DBus connection unique name or `org.qemu` as a fallback.
 *
 * Returns: A unique name the session is connected to
 */
const char *
mks_session_get_bus_name (MksSession *self)
{
  g_return_val_if_fail (MKS_IS_SESSION (self), NULL);

  return self->bus_name;
}

/**
 * mks_session_get_uuid:
 * @self: a #MksSession
 *
 * Gets the unique identifier of the VM.
 */
const char *
mks_session_get_uuid (MksSession *self)
{
  g_return_val_if_fail (MKS_IS_SESSION (self), NULL);

  return self->uuid;
}

/**
 * mks_session_get_name:
 * @self: a #MksSession
 *
 * Gets the name of the VM.
 */
const char *
mks_session_get_name (MksSession *self)
{
  g_return_val_if_fail (MKS_IS_SESSION (self), NULL);

  return self->name;
}

/**
 * mks_session_ref_screen:
 * @self: a #MksSession
 *
 * Gets the main screen for the session.
 *
 * Returns: (transfer full) (nullable): a #MksScreen or %NULL
 */
MksScreen *
mks_session_ref_screen (MksSession *self)
{
  GListModel *model;
  guint n_items;

  g_return_val_if_fail (MKS_IS_SESSION (self), NULL);

  model = G_LIST_MODEL (self->devices);
  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(MksDevice) device = g_list_model_get_item (model, i);

      if (MKS_IS_SCREEN (device))
        return MKS_SCREEN (g_steal_pointer (&device));
    }

  return NULL;
}
