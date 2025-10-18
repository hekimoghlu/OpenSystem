/* foundry-command-line-remote.c
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
#include "foundry-inhibitor-private.h"
#include "foundry-ipc.h"

struct _FoundryCommandLineRemote
{
  FoundryCommandLine   parent_instance;
  DexCancellable      *cancellable;
  FoundryContext      *context;
  FoundryInhibitor    *inhibitor;
  DexFuture           *proxy;
  GDBusConnection     *connection;
  char                *object_path;
  char                *directory;
  char               **env;
  int                  stdin_fd;
  int                  stdout_fd;
  int                  stderr_fd;
};

enum {
  PROP_0,
  PROP_CONNECTION,
  PROP_OBJECT_PATH,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryCommandLineRemote, foundry_command_line_remote, FOUNDRY_TYPE_COMMAND_LINE)

static GParamSpec *properties[N_PROPS];

static DexCancellable *
foundry_command_line_remote_dup_cancellable (FoundryCommandLine *command_line)
{
  return dex_ref (FOUNDRY_COMMAND_LINE_REMOTE (command_line)->cancellable);
}

static const char *
foundry_command_line_remote_getenv (FoundryCommandLine *command_line,
                                    const char         *name)
{
  FoundryCommandLineRemote *self = FOUNDRY_COMMAND_LINE_REMOTE (command_line);

  return g_environ_getenv (self->env, name);
}

static char **
foundry_command_line_remote_get_environ (FoundryCommandLine *command_line)
{
  FoundryCommandLineRemote *self = FOUNDRY_COMMAND_LINE_REMOTE (command_line);

  return g_strdupv (self->env);
}

static void
get_proxy_cb (GObject      *object,
              GAsyncResult *result,
              gpointer      user_data)
{
  g_autoptr(FoundryIpcCommandLine) proxy = NULL;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (!object || G_IS_OBJECT (object));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!(proxy = foundry_ipc_command_line_proxy_new_finish (result, &error)))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_object (promise, g_steal_pointer (&proxy));
}

static DexFuture *
get_proxy (FoundryCommandLineRemote *self)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE_REMOTE (self));

  if (self->proxy == NULL)
    {
      self->proxy = DEX_FUTURE (dex_promise_new ());
      foundry_ipc_command_line_proxy_new (self->connection,
                                          G_DBUS_PROXY_FLAGS_NONE,
                                          NULL,
                                          self->object_path,
                                          NULL,
                                          get_proxy_cb,
                                          dex_ref (self->proxy));
    }

  return dex_ref (self->proxy);
}

static void
call_open_cb (GObject      *object,
              GAsyncResult *result,
              gpointer      user_data)
{
  FoundryIpcCommandLine *proxy = (FoundryIpcCommandLine *)object;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GUnixFDList) fd_list = NULL;
  g_autoptr(GVariant) handle = NULL;
  g_autofd int fd = -1;

  g_assert (FOUNDRY_IPC_IS_COMMAND_LINE (proxy));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!foundry_ipc_command_line_call_open_finish (proxy, &handle, &fd_list, result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else if (-1 == (fd = g_unix_fd_list_get (fd_list, g_variant_get_handle (handle), &error)))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_fd (promise, g_steal_fd (&fd));
}

static DexFuture *
call_open (DexFuture *completed,
           gpointer   user_data)
{
  g_autoptr(FoundryIpcCommandLine) proxy = NULL;
  DexPromise *promise;
  int fd_number = GPOINTER_TO_INT (user_data);

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (fd_number >= 0);

  proxy = dex_await_object (dex_ref (completed), NULL);
  promise = dex_promise_new_cancellable ();

  foundry_ipc_command_line_call_open (proxy,
                                      fd_number,
                                      NULL,
                                      dex_promise_get_cancellable (promise),
                                      call_open_cb,
                                      dex_ref (promise));

  return DEX_FUTURE (promise);
}

static DexFuture *
foundry_command_line_remote_open (FoundryCommandLine *command_line,
                                  int                 fd_number)
{
  FoundryCommandLineRemote *self = (FoundryCommandLineRemote *)command_line;
  DexFuture *future;

  g_assert (FOUNDRY_IS_COMMAND_LINE_REMOTE (self));
  g_assert (fd_number >= 0);

  future = get_proxy (self);
  future = dex_future_then (future, call_open, GINT_TO_POINTER (fd_number), NULL);

  return future;
}

static gboolean
foundry_command_line_remote_isatty (FoundryCommandLine *command_line)
{
  FoundryCommandLineRemote *self = (FoundryCommandLineRemote *)command_line;

  g_assert (FOUNDRY_IS_COMMAND_LINE_REMOTE (self));

  return isatty (self->stdin_fd);
}

static void
write_all (int         fd,
           const char *message,
           gssize      to_write)
{
  const char *data = message;

  if (fd < 0)
    return;

  if (to_write < 0)
    to_write = strlen (message);

  while (to_write > 0)
    {
      gssize n_written;

      errno = 0;
      n_written = write (fd, data, to_write);

      if (n_written < 0)
        return;

      if (n_written == 0 && errno == EINTR)
        continue;

      to_write -= n_written;
      data += (gsize)n_written;
    }
}

static void
foundry_command_line_remote_print (FoundryCommandLine *command_line,
                                   const char         *message)
{
  write_all (FOUNDRY_COMMAND_LINE_REMOTE (command_line)->stdout_fd, message, -1);
}

static void
foundry_command_line_remote_printerr (FoundryCommandLine *command_line,
                                      const char         *message)
{
  write_all (FOUNDRY_COMMAND_LINE_REMOTE (command_line)->stderr_fd, message, -1);
}

static char *
foundry_command_line_remote_get_directory (FoundryCommandLine *command_line)
{
  return g_strdup (FOUNDRY_COMMAND_LINE_REMOTE (command_line)->directory);
}

static int
foundry_command_line_remote_get_stdin (FoundryCommandLine *self)
{
  return FOUNDRY_COMMAND_LINE_REMOTE (self)->stdin_fd;
}

static int
foundry_command_line_remote_get_stdout (FoundryCommandLine *self)
{
  return FOUNDRY_COMMAND_LINE_REMOTE (self)->stdout_fd;
}

static int
foundry_command_line_remote_get_stderr (FoundryCommandLine *self)
{
  return FOUNDRY_COMMAND_LINE_REMOTE (self)->stderr_fd;
}

static void
foundry_command_line_remote_finalize (GObject *object)
{
  FoundryCommandLineRemote *self = (FoundryCommandLineRemote *)object;

  dex_clear (&self->proxy);
  dex_clear (&self->cancellable);

  g_clear_object (&self->connection);
  g_clear_object (&self->context);
  g_clear_object (&self->inhibitor);

  g_clear_pointer (&self->object_path, g_free);
  g_clear_pointer (&self->directory, g_free);

  g_clear_pointer (&self->env, g_strfreev);

  g_clear_fd (&self->stdin_fd, NULL);
  g_clear_fd (&self->stdout_fd, NULL);
  g_clear_fd (&self->stderr_fd, NULL);

  G_OBJECT_CLASS (foundry_command_line_remote_parent_class)->finalize (object);
}

static void
foundry_command_line_remote_get_property (GObject    *object,
                                          guint       prop_id,
                                          GValue     *value,
                                          GParamSpec *pspec)
{
  FoundryCommandLineRemote *self = FOUNDRY_COMMAND_LINE_REMOTE (object);

  switch (prop_id)
    {
    case PROP_CONNECTION:
      g_value_set_object (value, self->connection);
      break;

    case PROP_OBJECT_PATH:
      g_value_set_string (value, self->object_path);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_command_line_remote_set_property (GObject      *object,
                                          guint         prop_id,
                                          const GValue *value,
                                          GParamSpec   *pspec)
{
  FoundryCommandLineRemote *self = FOUNDRY_COMMAND_LINE_REMOTE (object);

  switch (prop_id)
    {
    case PROP_CONNECTION:
      self->connection = g_value_dup_object (value);
      break;

    case PROP_OBJECT_PATH:
      self->object_path = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_command_line_remote_class_init (FoundryCommandLineRemoteClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryCommandLineClass *command_line_class = FOUNDRY_COMMAND_LINE_CLASS (klass);

  object_class->finalize = foundry_command_line_remote_finalize;
  object_class->get_property = foundry_command_line_remote_get_property;
  object_class->set_property = foundry_command_line_remote_set_property;

  command_line_class->get_directory = foundry_command_line_remote_get_directory;
  command_line_class->getenv = foundry_command_line_remote_getenv;
  command_line_class->get_environ = foundry_command_line_remote_get_environ;
  command_line_class->open = foundry_command_line_remote_open;
  command_line_class->isatty = foundry_command_line_remote_isatty;
  command_line_class->print = foundry_command_line_remote_print;
  command_line_class->printerr = foundry_command_line_remote_printerr;
  command_line_class->get_stdin = foundry_command_line_remote_get_stdin;
  command_line_class->get_stdout = foundry_command_line_remote_get_stdout;
  command_line_class->get_stderr = foundry_command_line_remote_get_stderr;
  command_line_class->dup_cancellable = foundry_command_line_remote_dup_cancellable;

  properties[PROP_CONNECTION] =
    g_param_spec_object ("connection", NULL, NULL,
                         G_TYPE_DBUS_CONNECTION,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_OBJECT_PATH] =
    g_param_spec_string ("object-path", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_command_line_remote_init (FoundryCommandLineRemote *self)
{
  self->stdin_fd = -1;
  self->stdout_fd = -1;
  self->stderr_fd = -1;

  self->cancellable = dex_cancellable_new ();
}

static void
foundry_command_line_remote_connection_closed (FoundryCommandLineRemote *self,
                                               gboolean                  remote_peer_vanished,
                                               const GError             *error,
                                               GDBusConnection          *connection)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE_REMOTE (self));
  g_assert (G_IS_DBUS_CONNECTION (connection));

  if (self->cancellable != NULL)
    dex_cancellable_cancel (self->cancellable);
}

FoundryCommandLine *
foundry_command_line_remote_new (FoundryContext     *context,
                                 const char         *directory,
                                 const char * const *env,
                                 int                 stdin_fd,
                                 int                 stdout_fd,
                                 int                 stderr_fd,
                                 GDBusConnection    *connection,
                                 const char         *object_path)
{
  FoundryCommandLineRemote *self;

  g_return_val_if_fail (!context || FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (directory != NULL, NULL);
  g_return_val_if_fail (G_IS_DBUS_CONNECTION (connection), NULL);
  g_return_val_if_fail (g_variant_is_object_path (object_path), NULL);

  self = g_object_new (FOUNDRY_TYPE_COMMAND_LINE_REMOTE,
                       "connection", connection,
                       "object-path", object_path,
                       NULL);

  g_signal_connect_object (connection,
                           "closed",
                           G_CALLBACK (foundry_command_line_remote_connection_closed),
                           self,
                           G_CONNECT_SWAPPED);

  if (g_set_object (&self->context, context))
    self->inhibitor = foundry_inhibitor_new (context, NULL);

  self->stdin_fd = g_steal_fd (&stdin_fd);
  self->stdout_fd = g_steal_fd (&stdout_fd);
  self->stderr_fd = g_steal_fd (&stderr_fd);
  self->directory = g_strdup (directory);
  self->env = g_strdupv ((char **)env);

  return FOUNDRY_COMMAND_LINE (self);
}

FoundryContext *
foundry_command_line_remote_get_context (FoundryCommandLineRemote *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE_REMOTE (self), NULL);

  return self->context;
}
