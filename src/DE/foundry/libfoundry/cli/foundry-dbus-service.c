/* foundry-dbus-service.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <glib/gstdio.h>

#include "foundry-command-line-remote-private.h"
#include "foundry-context.h"
#include "foundry-dbus-service.h"
#include "foundry-debug.h"
#include "foundry-directory-reaper.h"
#include "foundry-ipc.h"
#include "foundry-service-private.h"
#include "foundry-util-private.h"

struct _FoundryDBusService
{
  FoundryService  parent_instance;
  GDBusServer    *server;
  char           *address;
  char           *dbus_socket_dir;
  DexFuture      *run_fiber;
};

struct _FoundryDBusServiceClass
{
  FoundryServiceClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryDBusService, foundry_dbus_service, FOUNDRY_TYPE_SERVICE)

typedef struct _ProxiedRun
{
  FoundryIpcCommandLineService *service;
  GDBusMethodInvocation        *invocation;
} ProxiedRun;

static void
proxied_run_free (ProxiedRun *state)
{
  g_clear_object (&state->service);
  g_clear_object (&state->invocation);
  g_free (state);
}

static DexFuture *
foundry_dbus_service_run_complete_cb (DexFuture *completed,
                                      gpointer   user_data)
{
  ProxiedRun *state = user_data;
  g_autoptr(GError) error = NULL;
  int ret;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (state->invocation));
  g_assert (FOUNDRY_IPC_IS_COMMAND_LINE_SERVICE (state->service));

  ret = dex_await_int (dex_ref (completed), &error);

  if (error != NULL)
    g_dbus_method_invocation_return_gerror (g_object_ref (state->invocation), error);
  else
    foundry_ipc_command_line_service_complete_run (state->service,
                                                   g_object_ref (state->invocation),
                                                   NULL,
                                                   ret);

  return NULL;
}

static gboolean
foundry_dbus_service_handle_run_cb (FoundryIpcCommandLineService *service,
                                    GDBusMethodInvocation        *invocation,
                                    GUnixFDList                  *fd_list,
                                    const char                   *arg_directory,
                                    const char * const           *arg_environment,
                                    const char * const           *arg_argv,
                                    GVariant                     *arg_stdin_handle,
                                    GVariant                     *arg_stdout_handle,
                                    GVariant                     *arg_stderr_handle,
                                    const char                   *arg_object_path,
                                    FoundryDBusService           *self)
{
  g_autoptr(FoundryCommandLine) wrapped = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autofd int stdin_fd = -1;
  g_autofd int stdout_fd = -1;
  g_autofd int stderr_fd = -1;
  ProxiedRun *state;

  g_assert (FOUNDRY_IPC_IS_COMMAND_LINE_SERVICE (service));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (invocation));
  g_assert (G_IS_UNIX_FD_LIST (fd_list));
  g_assert (g_variant_is_object_path (arg_object_path));
  g_assert (FOUNDRY_IS_DBUS_SERVICE (self));

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      g_dbus_method_invocation_return_gerror (g_steal_pointer (&invocation),
                                              g_error_new (G_IO_ERROR,
                                                           G_IO_ERROR_CLOSED,
                                                           "Context is closed"));
      return TRUE;
    }

  state = g_new0 (ProxiedRun, 1);
  state->invocation = g_object_ref (invocation);
  state->service = g_object_ref (service);

  stdin_fd = g_unix_fd_list_get (fd_list, g_variant_get_handle (arg_stdin_handle), NULL);
  stdout_fd = g_unix_fd_list_get (fd_list, g_variant_get_handle (arg_stdout_handle), NULL);
  stderr_fd = g_unix_fd_list_get (fd_list, g_variant_get_handle (arg_stderr_handle), NULL);

  wrapped = foundry_command_line_remote_new (context,
                                             arg_directory,
                                             arg_environment,
                                             g_steal_fd (&stdin_fd),
                                             g_steal_fd (&stdout_fd),
                                             g_steal_fd (&stderr_fd),
                                             g_dbus_method_invocation_get_connection (invocation),
                                             arg_object_path);

  dex_future_disown (dex_future_finally (foundry_command_line_run (wrapped, arg_argv),
                                         foundry_dbus_service_run_complete_cb,
                                         state,
                                         (GDestroyNotify)proxied_run_free));

  return TRUE;
}

static void
foundry_dbus_service_connection_closed_cb (FoundryDBusService *self,
                                           gboolean            remote_peer_vanished,
                                           const GError       *error,
                                           GDBusConnection    *connection)
{
  g_assert (FOUNDRY_IS_DBUS_SERVICE (self));
  g_assert (G_IS_DBUS_CONNECTION (connection));

  g_signal_handlers_disconnect_by_func (connection,
                                        G_CALLBACK (foundry_dbus_service_connection_closed_cb),
                                        self);

  g_object_unref (connection);
}

static gboolean
foundry_dbus_service_handle_new_connection_cb (FoundryDBusService *self,
                                               GDBusConnection    *connection,
                                               GDBusServer        *server)
{
  g_autoptr(FoundryIpcCommandLineService) service = NULL;

  g_assert (FOUNDRY_IS_DBUS_SERVICE (self));
  g_assert (G_IS_DBUS_CONNECTION (connection));
  g_assert (G_IS_DBUS_SERVER (server));

  service = foundry_ipc_command_line_service_skeleton_new ();
  g_signal_connect_object (service,
                           "handle-run",
                           G_CALLBACK (foundry_dbus_service_handle_run_cb),
                           self,
                           0);
  g_dbus_interface_skeleton_export (G_DBUS_INTERFACE_SKELETON (service),
                                    connection,
                                    "/app/devsuite/foundry/CommandLine",
                                    NULL);
  g_object_set_data_full (G_OBJECT (connection),
                          "FOUNDRY_IPC_COMMAND_LINE",
                          g_steal_pointer (&service),
                          g_object_unref);

  g_object_ref (connection);
  g_signal_connect_object (connection,
                           "closed",
                           G_CALLBACK (foundry_dbus_service_connection_closed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  return TRUE;
}

static DexFuture *
foundry_dbus_service_run_fiber (gpointer user_data)
{
  static const char *dbus_socket_template = "foundry-dbus-XXXXXX";
  FoundryDBusService *self = user_data;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GDBusServer) server = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) state_directory = NULL;
  g_autoptr(GFile) tmpdir = NULL;
  g_autofree char *dbus_socket_dir = NULL;
  g_autofree char *guid = NULL;
  g_autofree char *address = NULL;

  g_assert (FOUNDRY_IS_DBUS_SERVICE (self));

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_CANCELLED,
                                  "Operation cancelled");

  state_directory = foundry_context_dup_state_directory (context);
  tmpdir = g_file_get_child (state_directory, "tmp");

  /* Ignore failure whether it is exists or otherwise, we'll catch the
   * error again when creating our dbus tmpdir.
   */
  dex_await (dex_file_make_directory (tmpdir, G_PRIORITY_DEFAULT), NULL);

  /* Try to create our temporary directory for DBus server for which clients
   * will connect to get access to this context instance.
   */
  if (!(dbus_socket_dir = dex_await_string (_foundry_mkdtemp (g_file_peek_path (tmpdir),
                                                              dbus_socket_template),
                                            &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* We will need to remove this at shutdown */
  g_set_str (&self->dbus_socket_dir, dbus_socket_dir);

  /* Create our D-Bus server for clients to connect to using the
   * socket in our foundry-specific temporary directory.
   */
  guid = g_dbus_generate_guid ();
  address = g_strdup_printf ("unix:tmpdir=%s", dbus_socket_dir);
  if (!(server = g_dbus_server_new_sync (address,
                                         G_DBUS_SERVER_FLAGS_AUTHENTICATION_ALLOW_ANONYMOUS,
                                         guid,
                                         NULL,
                                         NULL,
                                         &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  /* Keep the address around for use by process launchers */
  g_set_str (&self->address, g_dbus_server_get_client_address (server));

  g_signal_connect_object (server,
                           "new-connection",
                           G_CALLBACK (foundry_dbus_service_handle_new_connection_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_dbus_server_start (server);

  return dex_future_new_take_string (g_steal_pointer (&address));
}

/**
 * foundry_dbus_service_query_address:
 * @self: a [class@Foundry.DBusService]
 *
 * Ensures the GDBusServer is running and provides the address.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a string containing the address, or rejects with error.
 *
 * Since: 1.1
 */
DexFuture *
foundry_dbus_service_query_address (FoundryDBusService *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DBUS_SERVICE (self));

  if (self->run_fiber == NULL)
    {
      self->run_fiber = dex_scheduler_spawn (NULL, 0,
                                             foundry_dbus_service_run_fiber,
                                             g_object_ref (self),
                                             g_object_unref);
      dex_future_disown (dex_ref (self->run_fiber));
    }

  return dex_ref (self->run_fiber);
}

static DexFuture *
foundry_dbus_service_stop (FoundryService *service)
{
  FoundryDBusService *self = (FoundryDBusService *)service;
  g_autoptr(FoundryDirectoryReaper) reaper = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) directory = NULL;

  g_assert (FOUNDRY_IS_DBUS_SERVICE (self));

  dex_clear (&self->run_fiber);

  if (self->server != NULL)
    {
      g_dbus_server_stop (self->server);
      g_clear_object (&self->server);
    }

  if (self->dbus_socket_dir == NULL)
    return dex_future_new_true ();

  directory = g_file_new_for_path (self->dbus_socket_dir);
  reaper = foundry_directory_reaper_new ();
  foundry_directory_reaper_add_directory (reaper, directory, 0);
  foundry_directory_reaper_add_file (reaper, directory, 0);

  g_clear_pointer (&self->dbus_socket_dir, g_free);

  return foundry_directory_reaper_execute (reaper);
}

static void
foundry_dbus_service_finalize (GObject *object)
{
  FoundryDBusService *self = (FoundryDBusService *)object;

  dex_clear (&self->run_fiber);
  g_clear_object (&self->server);
  g_clear_pointer (&self->dbus_socket_dir, g_free);
  g_clear_pointer (&self->address, g_free);

  G_OBJECT_CLASS (foundry_dbus_service_parent_class)->finalize (object);
}

static void
foundry_dbus_service_class_init (FoundryDBusServiceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->finalize = foundry_dbus_service_finalize;

  service_class->stop = foundry_dbus_service_stop;
}

static void
foundry_dbus_service_init (FoundryDBusService *self)
{
}

/**
 * foundry_dbus_service_dup_address:
 * @self: a #FoundryDBusService
 *
 * Gets the D-Bus address of the embedded D-Bus server.
 *
 * Returns: (transfer full) (nullable): A D-Bus server address if started
 *   successfully otherwise %NULL.
 *
 * Deprecated: 1.1
 */
char *
foundry_dbus_service_dup_address (FoundryDBusService *self)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (FOUNDRY_IS_DBUS_SERVICE (self), NULL);

  return g_strdup (self->address);
}
