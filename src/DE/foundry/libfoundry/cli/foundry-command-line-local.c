/* foundry-command-line-local.c
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

#include <errno.h>
#include <unistd.h>

#include <glib/gi18n-lib.h>
#include <glib/gstdio.h>

#include "foundry-cli-command-tree.h"
#include "foundry-command-line-local-private.h"
#include "foundry-init.h"
#include "foundry-ipc.h"
#include "foundry-util-private.h"

struct _FoundryCommandLineLocal
{
  FoundryCommandLine     parent_instance;
  FoundryIpcCommandLine *skeleton;
  char                  *object_path;
};

G_DEFINE_FINAL_TYPE (FoundryCommandLineLocal, foundry_command_line_local, FOUNDRY_TYPE_COMMAND_LINE)

static void
bus_new_for_address_cb (GObject      *object,
                        GAsyncResult *result,
                        gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  GDBusConnection *conn;
  GError *error = NULL;

  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!(conn = g_dbus_connection_new_for_address_finish (result, &error)))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_object (promise, g_steal_pointer (&conn));
}

static DexFuture *
bus_new_for_address (const char           *address,
                     GDBusConnectionFlags  flags,
                     GDBusAuthObserver*    observer)
{
  DexPromise *promise;

  g_return_val_if_fail (address != NULL, NULL);
  g_return_val_if_fail (!observer || G_IS_DBUS_AUTH_OBSERVER (observer), NULL);

  promise = dex_promise_new_cancellable ();

  g_dbus_connection_new_for_address (address,
                                     flags,
                                     observer,
                                     dex_promise_get_cancellable (promise),
                                     bus_new_for_address_cb,
                                     dex_ref (promise));

  return DEX_FUTURE (promise);
}

static void
command_line_service_proxy_new_cb (GObject      *object,
                                   GAsyncResult *result,
                                   gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  FoundryIpcCommandLineService *ret;
  GError *error = NULL;

  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!(ret = foundry_ipc_command_line_service_proxy_new_finish (result, &error)))
    {
      dex_promise_reject (promise, g_steal_pointer (&error));
      return;
    }

  g_object_set (ret,
                "g-default-timeout", G_MAXINT,
                NULL);

  dex_promise_resolve_object (promise, g_steal_pointer (&ret));
}

static DexFuture *
command_line_service_proxy_new (GDBusConnection *conn,
                                GDBusProxyFlags  flags,
                                const char      *name,
                                const char      *object_path)
{
  DexPromise *promise;

  g_return_val_if_fail (G_IS_DBUS_CONNECTION (conn), NULL);

  promise = dex_promise_new_cancellable ();

  foundry_ipc_command_line_service_proxy_new (conn,
                                              flags,
                                              name,
                                              object_path,
                                              dex_promise_get_cancellable (promise),
                                              command_line_service_proxy_new_cb,
                                              dex_ref (promise));

  return DEX_FUTURE (promise);
}

static void
command_line_service_proxy_call_run_cb (GObject      *object,
                                        GAsyncResult *result,
                                        gpointer      user_data)
{
  FoundryIpcCommandLineService *proxy = (FoundryIpcCommandLineService *)object;
  g_autoptr(DexPromise) promise = user_data;
  GError *error = NULL;
  int ret = EXIT_FAILURE;

  g_assert (FOUNDRY_IPC_IS_COMMAND_LINE_SERVICE (proxy));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!foundry_ipc_command_line_service_call_run_finish (proxy, &ret, NULL, result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_int (promise, ret);
}

static DexFuture *
command_line_service_proxy_call_run (FoundryIpcCommandLineService *proxy,
                                     const char                   *directory,
                                     const char * const           *environ,
                                     const char * const           *argv,
                                     GVariant                     *stdin_handle,
                                     GVariant                     *stdout_handle,
                                     GVariant                     *stderr_handle,
                                     const char                   *object_path,
                                     GUnixFDList                  *fd_list)
{
  DexPromise *promise;

  g_return_val_if_fail (FOUNDRY_IPC_IS_COMMAND_LINE_SERVICE (proxy), NULL);

  promise = dex_promise_new_cancellable ();

  foundry_ipc_command_line_service_call_run (proxy,
                                             directory,
                                             environ,
                                             argv,
                                             stdin_handle,
                                             stdout_handle,
                                             stderr_handle,
                                             object_path,
                                             fd_list,
                                             dex_promise_get_cancellable (promise),
                                             command_line_service_proxy_call_run_cb,
                                             dex_ref (promise));

  return DEX_FUTURE (promise);
}

static gboolean
foundry_command_line_local_handle_open (FoundryIpcCommandLine *ipc,
                                        GDBusMethodInvocation *invocation,
                                        GUnixFDList           *in_fd_list,
                                        int                    arg_fd)
{
  g_autoptr(GUnixFDList) out_fd_list = NULL;
  g_autoptr(GError) error = NULL;
  g_autofd int fd = -1;
  int handle;

  g_assert (FOUNDRY_IPC_IS_COMMAND_LINE (ipc));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (invocation));
  g_assert (!in_fd_list || G_IS_UNIX_FD_LIST (in_fd_list));

  out_fd_list = g_unix_fd_list_new ();
  handle = g_unix_fd_list_append (out_fd_list, arg_fd, &error);

  if (handle < 0)
    g_dbus_method_invocation_return_gerror (g_steal_pointer (&invocation), error);
  else
    foundry_ipc_command_line_complete_open (ipc,
                                            g_steal_pointer (&invocation),
                                            out_fd_list,
                                            g_variant_new_handle (handle));

  return TRUE;
}

static void
foundry_command_line_local_constructed (GObject *object)
{
  FoundryCommandLineLocal *self = (FoundryCommandLineLocal *)object;

  G_OBJECT_CLASS (foundry_command_line_local_parent_class)->constructed (object);

  self->skeleton = foundry_ipc_command_line_skeleton_new ();
  g_signal_connect_object (self->skeleton,
                           "handle-open",
                           G_CALLBACK (foundry_command_line_local_handle_open),
                           self,
                           0);
}

static DexFuture *
foundry_command_line_local_open (FoundryCommandLine *command_line,
                                 int                 fd_number)
{
  DexPromise *promise;
  int fd;

  if ((fd = dup (fd_number)) < 0)
    {
      int errsv = errno;
      return dex_future_new_reject (G_FILE_ERROR,
                                    g_file_error_from_errno (errsv),
                                    "%s",
                                    g_strerror (errsv));
    }

  promise = dex_promise_new ();
  dex_promise_resolve_fd (promise, fd);

  return DEX_FUTURE (promise);
}

static char **
foundry_command_line_local_get_environ (FoundryCommandLine *command_line)
{
  return g_get_environ ();
}

static const char *
foundry_command_line_local_getenv (FoundryCommandLine *command_line,
                                   const char         *name)
{
  return g_getenv (name);
}

static gboolean
foundry_command_line_local_isatty (FoundryCommandLine *command_line)
{
  return isatty (STDIN_FILENO);
}

static void
foundry_command_line_local_print (FoundryCommandLine *command_line,
                                  const char         *message)
{
  _foundry_fd_write_all (STDOUT_FILENO, message, -1);
}

static void
foundry_command_line_local_printerr (FoundryCommandLine *command_line,
                                     const char         *message)
{
  _foundry_fd_write_all (STDERR_FILENO, message, -1);
}

static char *
foundry_command_line_local_get_directory (FoundryCommandLine *command_line)
{
  return g_get_current_dir ();
}

static gboolean
foundry_command_line_local_export (FoundryCommandLineLocal  *self,
                                   GDBusConnection          *connection,
                                   GError                  **error)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE_LOCAL (self));
  g_assert (G_IS_DBUS_CONNECTION (connection));

  return g_dbus_interface_skeleton_export (G_DBUS_INTERFACE_SKELETON (self->skeleton),
                                           connection,
                                           self->object_path,
                                           error);
}

static void
foundry_command_line_local_unexport (FoundryCommandLineLocal *self,
                                     GDBusConnection         *connection)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE_LOCAL (self));
  g_assert (G_IS_DBUS_CONNECTION (connection));
}

static DexFuture *
foundry_command_line_local_run_fiber (FoundryCommandLineLocal *self,
                                      const char * const      *argv)
{
  FoundryCommandLineClass *klass;
  const char *address;

  g_assert (FOUNDRY_IS_COMMAND_LINE_LOCAL (self));
  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);

  /* If this is a completion request do that */
  if (g_strcmp0 (argv[1], "complete") == 0 &&
      g_strv_length ((char **)argv) == 5)
    {
      FoundryCliCommandTree *tree = foundry_cli_command_tree_get_default ();
      g_auto(GStrv) completions = NULL;
      int pos = atoi (argv[3]);

      if ((completions = foundry_cli_command_tree_complete (tree,
                                                            FOUNDRY_COMMAND_LINE (self),
                                                            argv[2],
                                                            pos,
                                                            argv[4])))
        {
          for (guint i = 0; completions[i]; i++)
            g_print ("%s\n", completions[i]);
        }

      return dex_future_new_for_int (EXIT_SUCCESS);
    }

  klass = FOUNDRY_COMMAND_LINE_CLASS (foundry_command_line_local_parent_class);

  if ((address = g_getenv ("FOUNDRY_ADDRESS")))
    {
      g_autofree char *cwd = foundry_command_line_get_directory (FOUNDRY_COMMAND_LINE (self));
      g_autoptr(FoundryIpcCommandLineService) proxy = NULL;
      g_autoptr(GDBusConnection) bus = NULL;
      g_autoptr(GUnixFDList) fd_list = NULL;
      g_autoptr(GError) error = NULL;
      g_auto(GStrv) environ = NULL;
      int stdin_handle = -1;
      int stdout_handle = -1;
      int stderr_handle = -1;
      int ret;

      if (!(bus = dex_await_object (bus_new_for_address (address, G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_CLIENT, NULL), &error)) ||
          !(proxy = dex_await_object (command_line_service_proxy_new (bus, 0, NULL, "/app/devsuite/foundry/CommandLine"), &error)) ||
          !foundry_command_line_local_export (self, bus, &error))
        goto chain_up;

      fd_list = g_unix_fd_list_new ();
      stdin_handle = g_unix_fd_list_append (fd_list, STDIN_FILENO, NULL);
      stdout_handle = g_unix_fd_list_append (fd_list, STDOUT_FILENO, NULL);
      stderr_handle = g_unix_fd_list_append (fd_list, STDERR_FILENO, NULL);

      environ = foundry_command_line_get_environ (FOUNDRY_COMMAND_LINE (self));

      ret = dex_await_int (command_line_service_proxy_call_run (proxy,
                                                                cwd,
                                                                (const char * const *)environ,
                                                                argv,
                                                                g_variant_new_handle (stdin_handle),
                                                                g_variant_new_handle (stdout_handle),
                                                                g_variant_new_handle (stderr_handle),
                                                                self->object_path,
                                                                fd_list),
                           &error);

      foundry_command_line_local_unexport (self, bus);

      if (error != NULL)
        {
          g_dbus_error_strip_remote_error (error);

          if (g_error_matches (error, FOUNDRY_COMMAND_LINE_ERROR, FOUNDRY_COMMAND_LINE_ERROR_RUN_LOCAL))
            goto chain_up;

          return dex_future_new_for_error (g_steal_pointer (&error));
        }

      return dex_future_new_for_int (ret);
    }

chain_up:
  /* First ensure that we've completed initializing */
  dex_await (foundry_init (), NULL);

  return klass->run (FOUNDRY_COMMAND_LINE (self), argv);
}

static DexFuture *
foundry_command_line_local_run (FoundryCommandLine *command_line,
                                const char * const *argv)
{
  FoundryCommandLineLocal *self = (FoundryCommandLineLocal *)command_line;

  g_assert (FOUNDRY_IS_COMMAND_LINE_LOCAL (self));
  g_assert (argv != NULL);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_command_line_local_run_fiber),
                                  2,
                                  FOUNDRY_TYPE_COMMAND_LINE, self,
                                  G_TYPE_STRV, argv);
}

static int
foundry_command_line_local_get_stdin (FoundryCommandLine *self)
{
  return STDIN_FILENO;
}

static int
foundry_command_line_local_get_stdout (FoundryCommandLine *self)
{
  return STDOUT_FILENO;
}

static int
foundry_command_line_local_get_stderr (FoundryCommandLine *self)
{
  return STDERR_FILENO;
}

static void
foundry_command_line_local_finalize (GObject *object)
{
  FoundryCommandLineLocal *self = (FoundryCommandLineLocal *)object;

  g_clear_object (&self->skeleton);
  g_clear_pointer (&self->object_path, g_free);

  G_OBJECT_CLASS (foundry_command_line_local_parent_class)->finalize (object);
}

static void
foundry_command_line_local_class_init (FoundryCommandLineLocalClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryCommandLineClass *command_line_class = FOUNDRY_COMMAND_LINE_CLASS (klass);

  object_class->constructed = foundry_command_line_local_constructed;
  object_class->finalize = foundry_command_line_local_finalize;

  command_line_class->get_directory = foundry_command_line_local_get_directory;
  command_line_class->get_environ = foundry_command_line_local_get_environ;
  command_line_class->getenv = foundry_command_line_local_getenv;
  command_line_class->isatty = foundry_command_line_local_isatty;
  command_line_class->open = foundry_command_line_local_open;
  command_line_class->print = foundry_command_line_local_print;
  command_line_class->printerr = foundry_command_line_local_printerr;
  command_line_class->run = foundry_command_line_local_run;
  command_line_class->get_stdin = foundry_command_line_local_get_stdin;
  command_line_class->get_stdout = foundry_command_line_local_get_stdout;
  command_line_class->get_stderr = foundry_command_line_local_get_stderr;
}

static void
foundry_command_line_local_init (FoundryCommandLineLocal *self)
{
  g_autofree char *guid = g_dbus_generate_guid ();
  self->object_path = g_strdup_printf ("/app/devsuite/foundry/CommandLine/%s", guid);
}

FoundryCommandLine *
foundry_command_line_local_new (void)
{
  return g_object_new (FOUNDRY_TYPE_COMMAND_LINE_LOCAL, NULL);
}
