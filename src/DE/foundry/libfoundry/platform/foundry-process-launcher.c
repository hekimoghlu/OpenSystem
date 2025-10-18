/* foundry-process-launcher.c
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

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <glib-unix.h>
#include <glib/gstdio.h>

#include "foundry-debug.h"
#include "foundry-path.h"
#include "foundry-process-launcher.h"
#include "foundry-shell.h"
#include "foundry-util-private.h"

typedef struct
{
  GList                          qlink;
  char                          *cwd;
  GArray                        *argv;
  GArray                        *env;
  FoundryUnixFDMap              *unix_fd_map;
  FoundryProcessLauncherHandler  handler;
  gpointer                       handler_data;
  GDestroyNotify                 handler_data_destroy;
} FoundryProcessLauncherLayer;

struct _FoundryProcessLauncher
{
  GObject                     parent_instance;
  GQueue                      layers;
  FoundryProcessLauncherLayer root;
  guint                       ended : 1;
  guint                       setup_tty : 1;
};

G_DEFINE_FINAL_TYPE (FoundryProcessLauncher, foundry_process_launcher, G_TYPE_OBJECT)

FoundryProcessLauncher *
foundry_process_launcher_new (void)
{
  return g_object_new (FOUNDRY_TYPE_PROCESS_LAUNCHER, NULL);
}

static void
copy_envvar_with_fallback (FoundryProcessLauncher *process_launcher,
                           const char * const     *environ_,
                           const char             *key,
                           const char             *fallback)
{
  const char *val;

  if ((val = g_environ_getenv ((char **)environ_, key)))
    foundry_process_launcher_setenv (process_launcher, key, val);
  else if (fallback != NULL)
    foundry_process_launcher_setenv (process_launcher, key, fallback);
}

/**
 * foundry_process_launcher_add_minimal_environment:
 * @self: a #FoundryProcessLauncher
 *
 * Adds a minimal set of environment variables.
 *
 * This is useful to get access to things like the display or other
 * expected variables.
 */
void
foundry_process_launcher_add_minimal_environment (FoundryProcessLauncher *self)
{
  const gchar * const *host_environ = _foundry_host_environ ();
  static const char *copy_env[] = {
    "AT_SPI_BUS_ADDRESS",
    "DBUS_SESSION_BUS_ADDRESS",
    "DBUS_SYSTEM_BUS_ADDRESS",
    "DESKTOP_SESSION",
    "DISPLAY",
    "LANG",
    "HOME",
    "SHELL",
    "SSH_AUTH_SOCK",
    "USER",
    "WAYLAND_DISPLAY",
    "XAUTHORITY",
    "XDG_CURRENT_DESKTOP",
    "XDG_MENU_PREFIX",
    "XDG_SEAT",
    "XDG_SESSION_DESKTOP",
    "XDG_SESSION_ID",
    "XDG_SESSION_TYPE",
    "XDG_VTNR",
  };
  const char *val;

  FOUNDRY_ENTRY;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  for (guint i = 0; i < G_N_ELEMENTS (copy_env); i++)
    {
      const char *key = copy_env[i];

      if ((val = g_environ_getenv ((char **)host_environ, key)))
        foundry_process_launcher_setenv (self, key, val);
    }

  copy_envvar_with_fallback (self, host_environ, "TERM", "xterm-256color");
  copy_envvar_with_fallback (self, host_environ, "COLORTERM", "truecolor");

  FOUNDRY_EXIT;
}

static void
foundry_process_launcher_layer_clear (FoundryProcessLauncherLayer *layer)
{
  g_assert (layer != NULL);
  g_assert (layer->qlink.data == layer);
  g_assert (layer->qlink.prev == NULL);
  g_assert (layer->qlink.next == NULL);

  if (layer->handler_data_destroy)
    g_clear_pointer (&layer->handler_data, layer->handler_data_destroy);

  g_clear_pointer (&layer->cwd, g_free);
  g_clear_pointer (&layer->argv, g_array_unref);
  g_clear_pointer (&layer->env, g_array_unref);
  g_clear_object (&layer->unix_fd_map);
}

static void
foundry_process_launcher_layer_free (FoundryProcessLauncherLayer *layer)
{
  foundry_process_launcher_layer_clear (layer);

  g_slice_free (FoundryProcessLauncherLayer, layer);
}

static void
strptr_free (gpointer data)
{
  char **strptr = data;
  g_clear_pointer (strptr, g_free);
}

static void
foundry_process_launcher_layer_init (FoundryProcessLauncherLayer *layer)
{
  g_assert (layer != NULL);

  layer->qlink.data = layer;
  layer->argv = g_array_new (TRUE, TRUE, sizeof (char *));
  layer->env = g_array_new (TRUE, TRUE, sizeof (char *));
  layer->unix_fd_map = foundry_unix_fd_map_new ();

  g_array_set_clear_func (layer->argv, strptr_free);
  g_array_set_clear_func (layer->env, strptr_free);
}

static FoundryProcessLauncherLayer *
foundry_process_launcher_current_layer (FoundryProcessLauncher *self)
{
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_assert (self->layers.length > 0);

  return self->layers.head->data;
}

static void
foundry_process_launcher_dispose (GObject *object)
{
  FoundryProcessLauncher *self = (FoundryProcessLauncher *)object;
  FoundryProcessLauncherLayer *layer;

  while ((layer = g_queue_peek_head (&self->layers)))
    {
      g_queue_unlink (&self->layers, &layer->qlink);
      if (layer != &self->root)
        foundry_process_launcher_layer_free (layer);
    }

  foundry_process_launcher_layer_clear (&self->root);

  G_OBJECT_CLASS (foundry_process_launcher_parent_class)->dispose (object);
}

static void
foundry_process_launcher_class_init (FoundryProcessLauncherClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_process_launcher_dispose;
}

static void
foundry_process_launcher_init (FoundryProcessLauncher *self)
{
  foundry_process_launcher_layer_init (&self->root);

  g_queue_push_head_link (&self->layers, &self->root.qlink);

  self->setup_tty = TRUE;
}

void
foundry_process_launcher_push (FoundryProcessLauncher        *self,
                               FoundryProcessLauncherHandler  handler,
                               gpointer                       handler_data,
                               GDestroyNotify                 handler_data_destroy)
{
  FoundryProcessLauncherLayer *layer;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  layer = g_slice_new0 (FoundryProcessLauncherLayer);

  foundry_process_launcher_layer_init (layer);

  layer->handler = handler;
  layer->handler_data = handler_data;
  layer->handler_data_destroy = handler_data_destroy;

  g_queue_push_head_link (&self->layers, &layer->qlink);
}

void
foundry_process_launcher_push_at_base (FoundryProcessLauncher        *self,
                                       FoundryProcessLauncherHandler  handler,
                                       gpointer                       handler_data,
                                       GDestroyNotify                 handler_data_destroy)
{
  FoundryProcessLauncherLayer *layer;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  layer = g_slice_new0 (FoundryProcessLauncherLayer);

  foundry_process_launcher_layer_init (layer);

  layer->handler = handler;
  layer->handler_data = handler_data;
  layer->handler_data_destroy = handler_data_destroy;

  g_queue_insert_before_link (&self->layers, &self->root.qlink, &layer->qlink);
}

static gboolean
foundry_process_launcher_host_handler (FoundryProcessLauncher  *self,
                                       const char * const      *argv,
                                       const char * const      *env,
                                       const char              *cwd,
                                       FoundryUnixFDMap        *unix_fd_map,
                                       gpointer                 user_data,
                                       GError                 **error)
{
  const char *dbus_session_bus_address;
  guint length;

  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_assert (argv != NULL);
  g_assert (env != NULL);
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (unix_fd_map));
  g_assert (_foundry_in_container ());

  /* Make sure we can access the right D-Bus for auto-spawn */
  if ((dbus_session_bus_address = g_getenv ("DBUS_SESSION_BUS_ADDRESS")))
    foundry_process_launcher_setenv (self, "DBUS_SESSION_BUS_ADDRESS", dbus_session_bus_address);

  foundry_process_launcher_append_argv (self, "flatpak-spawn");
  foundry_process_launcher_append_argv (self, "--host");
  foundry_process_launcher_append_argv (self, "--watch-bus");

  if (env != NULL)
    {
      for (guint i = 0; env[i]; i++)
        foundry_process_launcher_append_formatted (self, "--env=%s", env[i]);
    }

  if (cwd != NULL)
    foundry_process_launcher_append_formatted (self, "--directory=%s", cwd);

  if ((length = foundry_unix_fd_map_get_length (unix_fd_map)))
    {
      for (guint i = 0; i < length; i++)
        {
          int source_fd;
          int dest_fd;

          source_fd = foundry_unix_fd_map_peek (unix_fd_map, i, &dest_fd);

          if (dest_fd < STDERR_FILENO)
            continue;

          g_debug ("Mapping Builder FD %d to target FD %d via flatpak-spawn",
                   source_fd, dest_fd);

          if (source_fd != -1 && dest_fd != -1)
            foundry_process_launcher_append_formatted (self, "--forward-fd=%d", dest_fd);
        }

      if (!foundry_process_launcher_merge_unix_fd_map (self, unix_fd_map, error))
        return FALSE;
    }

  /* Now append the arguments */
  foundry_process_launcher_append_args (self, argv);

  return TRUE;
}

static gboolean
is_empty (FoundryProcessLauncher *self)
{
  FoundryProcessLauncherLayer *root;

  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  if (self->layers.length > 1)
    return FALSE;

  root = g_queue_peek_head (&self->layers);

  return root->argv->len == 0;
}

/**
 * foundry_process_launcher_push_host:
 * @self: a #FoundryProcessLauncher
 *
 * Pushes handler to transform command to run on host.
 *
 * If necessary, a layer is pushed to ensure the command is run on the
 * host instead of the application container.
 *
 * If Builder is running on the host already, this function does nothing.
 */
void
foundry_process_launcher_push_host (FoundryProcessLauncher *self)
{
  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  /* We use flatpak-spawn to jump to the host even if we're
   * inside a container like toolbox first.
   */
  if (_foundry_in_container () || !is_empty (self))
    {
      self->setup_tty = FALSE;
      foundry_process_launcher_push (self,
                                     foundry_process_launcher_host_handler,
                                     NULL,
                                     NULL);
    }
  else if (is_empty (self))
    {
      g_auto(GStrv) environ_ = g_get_environ ();

      environ_ = g_environ_unsetenv (environ_, "G_MESSAGES_DEBUG");

      /* If we're empty, act like we're already the host and ensure
       * that we get some environment variables to make things work.
       */
      foundry_process_launcher_set_environ (self, (const char * const *)environ_);
    }
}

typedef struct
{
  char *shell;
  FoundryProcessLauncherShell kind : 2;
} Shell;

static void
shell_free (gpointer data)
{
  Shell *shell = data;
  g_clear_pointer (&shell->shell, g_free);
  g_slice_free (Shell, shell);
}

static gboolean
is_simple_variable_expansion (const char *str)
{
  gboolean has_curly;

  g_assert (str != NULL);

  if (str[0] != '$')
    return FALSE;

  str++;

  if ((has_curly = str[0] == '{'))
    str++;

  if (!g_ascii_isalpha (str[0]))
    return FALSE;

  while (str[0] != 0)
    {
      if (has_curly && str[0] == '}')
        return TRUE;

      if (!g_ascii_isalnum (str[0]))
        return FALSE;

      str++;
    }

  return TRUE;
}

static gboolean
foundry_process_launcher_shell_handler (FoundryProcessLauncher  *self,
                                        const char * const      *argv,
                                        const char * const      *env,
                                        const char              *cwd,
                                        FoundryUnixFDMap        *unix_fd_map,
                                        gpointer                 user_data,
                                        GError                 **error)
{
  Shell *shell = user_data;
  g_autoptr(GString) str = NULL;

  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_assert (argv != NULL);
  g_assert (env != NULL);
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (unix_fd_map));
  g_assert (shell != NULL);
  g_assert (shell->shell != NULL);

  if (!foundry_process_launcher_merge_unix_fd_map (self, unix_fd_map, error))
    return FALSE;

  if (cwd != NULL)
    foundry_process_launcher_set_cwd (self, cwd);

  foundry_process_launcher_append_argv (self, shell->shell);
  if (shell->kind == FOUNDRY_PROCESS_LAUNCHER_SHELL_LOGIN)
    foundry_process_launcher_append_argv (self, "-l");
  else if (shell->kind == FOUNDRY_PROCESS_LAUNCHER_SHELL_INTERACTIVE)
    foundry_process_launcher_append_argv (self, "-i");
  foundry_process_launcher_append_argv (self, "-c");

  str = g_string_new (NULL);

  if (env[0] != NULL)
    {
      g_string_append (str, "env");

      for (guint i = 0; env[i]; i++)
        {
          g_autofree char *quoted = g_shell_quote (env[i]);

          g_string_append_c (str, ' ');
          g_string_append (str, quoted);
        }

      g_string_append_c (str, ' ');
    }

  for (guint i = 0; argv[i]; i++)
    {
      if (i > 0)
        g_string_append_c (str, ' ');

      if (is_simple_variable_expansion (argv[i]))
        {
          g_string_append (str, argv[i]);
        }
      else
        {
          g_autofree char *quoted = g_shell_quote (argv[i]);
          g_string_append (str, quoted);
        }
    }

  foundry_process_launcher_append_argv (self, str->str);

  return TRUE;
}

/**
 * foundry_process_launcher_push_shell:
 * @self: a #FoundryProcessLauncher
 * @shell: the kind of shell to be used
 *
 * Pushes a shell which can run the upper layer command with -c.
 */
void
foundry_process_launcher_push_shell (FoundryProcessLauncher      *self,
                                     FoundryProcessLauncherShell  shell)
{
  Shell *state;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  state = g_slice_new0 (Shell);
  state->shell = g_strdup ("/bin/sh");
  state->kind = shell;

  foundry_process_launcher_push (self, foundry_process_launcher_shell_handler, state, shell_free);
}

void
foundry_process_launcher_push_user_shell (FoundryProcessLauncher      *self,
                                          FoundryProcessLauncherShell  shell)
{
  const char *user_shell;
  Shell *state;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  user_shell = foundry_shell_get_default ();

  if (!foundry_shell_supports_dash_c (user_shell))
    user_shell = "/bin/sh";

  switch (shell)
    {
    default:
    case FOUNDRY_PROCESS_LAUNCHER_SHELL_DEFAULT:
      break;

    case FOUNDRY_PROCESS_LAUNCHER_SHELL_LOGIN:
      if (!foundry_shell_supports_dash_login (user_shell))
        user_shell = "/bin/sh";
      break;

    case FOUNDRY_PROCESS_LAUNCHER_SHELL_INTERACTIVE:
      break;
    }

  state = g_slice_new0 (Shell);
  state->shell = g_strdup (user_shell);
  state->kind = shell;

  foundry_process_launcher_push (self, foundry_process_launcher_shell_handler, state, shell_free);
}

static gboolean
foundry_process_launcher_error_handler (FoundryProcessLauncher  *self,
                                        const char * const      *argv,
                                        const char * const      *env,
                                        const char              *cwd,
                                        FoundryUnixFDMap        *unix_fd_map,
                                        gpointer                 user_data,
                                        GError                 **error)
{
  const GError *local_error = user_data;

  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (unix_fd_map));
  g_assert (local_error != NULL);

  if (error != NULL)
    *error = g_error_copy (local_error);

  return FALSE;
}

/**
 * foundry_process_launcher_push_error:
 * @self: a #FoundryProcessLauncher
 * @error: (transfer full) (in): a #GError
 *
 * Pushes a new layer that will always fail with @error.
 *
 * This is useful if you have an error when attempting to build
 * a run command, but need it to deliver the error when attempting
 * to create a subprocess launcher.
 */
void
foundry_process_launcher_push_error (FoundryProcessLauncher *self,
                                     GError                 *error)
{
  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_return_if_fail (error != NULL);

  foundry_process_launcher_push (self,
                                 foundry_process_launcher_error_handler,
                                 error,
                                 (GDestroyNotify)g_error_free);
}

static gboolean
next_variable (const char *str,
               guint      *cursor,
               guint      *begin)
{
  for (guint i = *cursor; str[i]; i++)
    {
      /* Skip past escaped $ */
      if (str[i] == '\\' && str[i+1] == '$')
        {
          i++;
          continue;
        }

      if (str[i] == '$')
        {
          guint j;

          *begin = i;
          *cursor = i;

          for (j = i+1; str[j]; j++)
            {
              if (!g_ascii_isalnum (str[j]) && str[j] != '_')
                {
                  *cursor = j;
                  break;
                }
            }

          if (str[j] == 0)
            *cursor = j;

          if (*cursor > ((*begin) + 1))
            return TRUE;
        }
    }

  return FALSE;
}

static char *
wordexp_with_environ (const char         *input,
                      const char * const *environ_)
{
  g_autoptr(GString) str = NULL;
  guint cursor = 0;
  guint begin;

  g_assert (input != NULL);
  g_assert (environ_ != NULL);

  str = g_string_new (input);

  while (next_variable (str->str, &cursor, &begin))
    {
      g_autofree char *key = NULL;
      guint key_len = cursor - begin;
      const char *value;
      guint value_len;

      g_assert (str->str[begin] == '$');

      key = g_strndup (str->str + begin, key_len);
      value = g_environ_getenv ((char **)environ_, key+1);
      value_len = value ? strlen (value) : 0;

      if (value != NULL)
        {
          g_string_erase (str, begin, key_len);
          g_string_insert_len (str, begin, value, value_len);

          if (value_len > key_len)
            cursor += (value_len - key_len);
          else if (value_len < key_len)
            cursor -= (key_len - value_len);
        }
    }

  return g_string_free (g_steal_pointer (&str), FALSE);
}

static gboolean
foundry_process_launcher_expansion_handler (FoundryProcessLauncher  *self,
                                            const char * const      *argv,
                                            const char * const      *env,
                                            const char              *cwd,
                                            FoundryUnixFDMap        *unix_fd_map,
                                            gpointer                 user_data,
                                            GError                 **error)
{
  const char * const *environ_ = user_data;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_assert (argv != NULL);
  g_assert (environ_ != NULL);
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (unix_fd_map));

  if (!foundry_process_launcher_merge_unix_fd_map (self, unix_fd_map, error))
    FOUNDRY_RETURN (FALSE);

  if (cwd != NULL)
    {
      g_autofree char *newcwd = wordexp_with_environ (cwd, environ_);
      g_autofree char *expanded = foundry_path_expand (newcwd);

      foundry_process_launcher_set_cwd (self, expanded);
    }

  if (env != NULL)
    {
      g_autoptr(GPtrArray) newenv = g_ptr_array_new_null_terminated (0, g_free, TRUE);

      for (guint i = 0; env[i]; i++)
        {
          char *expanded = wordexp_with_environ (env[i], environ_);
          g_ptr_array_add (newenv, expanded);
        }

      foundry_process_launcher_add_environ (self, (const char * const *)(gpointer)newenv->pdata);
    }

  if (argv != NULL)
    {
      g_autoptr(GPtrArray) newargv = g_ptr_array_new_null_terminated (0, g_free, TRUE);

      for (guint i = 0; argv[i]; i++)
        {
          char *expanded = wordexp_with_environ (argv[i], environ_);
          g_ptr_array_add (newargv, expanded);
        }

      foundry_process_launcher_append_args (self, (const char * const *)(gpointer)newargv->pdata);
    }

  FOUNDRY_RETURN (TRUE);
}

/**
 * foundry_process_launcher_push_expansion:
 * @self: a #FoundryProcessLauncher
 *
 * Pushes a layer to expand known environment variables.
 *
 * The command argv and cwd will have `$FOO` style environment
 * variables expanded that are known. This can be useful to allow
 * things like `$BUILDDIR` be expanded at this layer.
 */
void
foundry_process_launcher_push_expansion (FoundryProcessLauncher *self,
                                         const char * const     *environ_)
{
  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  if (environ_ != NULL)
    foundry_process_launcher_push (self,
                                   foundry_process_launcher_expansion_handler,
                                   g_strdupv ((char **)environ_),
                                   (GDestroyNotify)g_strfreev);
}

const char * const *
foundry_process_launcher_get_argv (FoundryProcessLauncher *self)
{
  FoundryProcessLauncherLayer *layer;

  g_return_val_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self), NULL);

  layer = foundry_process_launcher_current_layer (self);

  return (const char * const *)(gpointer)layer->argv->data;
}

void
foundry_process_launcher_set_argv (FoundryProcessLauncher *self,
                                   const char * const     *argv)
{
  FoundryProcessLauncherLayer *layer;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  layer = foundry_process_launcher_current_layer (self);

  g_array_set_size (layer->argv, 0);

  if (argv != NULL)
    {
      char **copy = g_strdupv ((char **)argv);
      g_array_append_vals (layer->argv, copy, g_strv_length (copy));
      g_free (copy);
    }
}

const char * const *
foundry_process_launcher_get_environ (FoundryProcessLauncher *self)
{
  FoundryProcessLauncherLayer *layer;

  g_return_val_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self), NULL);

  layer = foundry_process_launcher_current_layer (self);

  return (const char * const *)(gpointer)layer->env->data;
}

void
foundry_process_launcher_set_environ (FoundryProcessLauncher *self,
                                      const char * const     *environ_)
{
  FoundryProcessLauncherLayer *layer;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  layer = foundry_process_launcher_current_layer (self);

  g_array_set_size (layer->env, 0);

  if (environ_ != NULL && environ_[0] != NULL)
    {
      char **copy = g_strdupv ((char **)environ_);
      g_array_append_vals (layer->env, copy, g_strv_length (copy));
      g_free (copy);
    }
}

void
foundry_process_launcher_add_environ (FoundryProcessLauncher *self,
                                      const char * const     *environ_)
{
  FoundryProcessLauncherLayer *layer;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  if (environ_ == NULL || environ_[0] == NULL)
    return;

  layer = foundry_process_launcher_current_layer (self);

  for (guint i = 0; environ_[i]; i++)
    {
      const char *pair = environ_[i];
      const char *eq = strchr (pair, '=');
      char **dest = NULL;
      gsize keylen;

      if (eq == NULL)
        continue;

      keylen = eq - pair;

      for (guint j = 0; j < layer->env->len; j++)
        {
          const char *ele = g_array_index (layer->env, const char *, j);

          if (strncmp (pair, ele, keylen) == 0 && ele[keylen] == '=')
            {
              dest = &g_array_index (layer->env, char *, j);
              break;
            }
        }

      if (dest == NULL)
        {
          g_array_set_size (layer->env, layer->env->len + 1);
          dest = &g_array_index (layer->env, char *, layer->env->len - 1);
        }

      g_clear_pointer (dest, g_free);
      *dest = g_strdup (pair);
    }
}

const char *
foundry_process_launcher_get_cwd (FoundryProcessLauncher *self)
{
  FoundryProcessLauncherLayer *layer;

  g_return_val_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self), NULL);

  layer = foundry_process_launcher_current_layer (self);

  return layer->cwd;
}

void
foundry_process_launcher_set_cwd (FoundryProcessLauncher *self,
                                  const char             *cwd)
{
  FoundryProcessLauncherLayer *layer;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  layer = foundry_process_launcher_current_layer (self);

  g_set_str (&layer->cwd, cwd);
}

void
foundry_process_launcher_prepend_argv (FoundryProcessLauncher *self,
                                       const char             *arg)
{
  FoundryProcessLauncherLayer *layer;
  char *copy;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_return_if_fail (arg != NULL);

  layer = foundry_process_launcher_current_layer (self);

  copy = g_strdup (arg);
  g_array_insert_val (layer->argv, 0, copy);
}

void
foundry_process_launcher_prepend_args (FoundryProcessLauncher *self,
                                       const char * const     *args)
{
  FoundryProcessLauncherLayer *layer;
  char **copy;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  if (args == NULL || args[0] == NULL)
    return;

  layer = foundry_process_launcher_current_layer (self);

  copy = g_strdupv ((char **)args);
  g_array_insert_vals (layer->argv, 0, copy, g_strv_length (copy));
  g_free (copy);
}

void
foundry_process_launcher_append_argv (FoundryProcessLauncher *self,
                                      const char             *arg)
{
  FoundryProcessLauncherLayer *layer;
  char *copy;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_return_if_fail (arg != NULL);

  layer = foundry_process_launcher_current_layer (self);

  copy = g_strdup (arg);
  g_array_append_val (layer->argv, copy);
}

void
foundry_process_launcher_append_formatted (FoundryProcessLauncher *self,
                                           const char             *format,
                                           ...)
{
  g_autofree char *arg = NULL;
  va_list args;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_return_if_fail (format != NULL);

  va_start (args, format);
  arg = g_strdup_vprintf (format, args);
  va_end (args);

  foundry_process_launcher_append_argv (self, arg);
}

void
foundry_process_launcher_append_args (FoundryProcessLauncher *self,
                                      const char * const     *args)
{
  FoundryProcessLauncherLayer *layer;
  char **copy;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  if (args == NULL || args[0] == NULL)
    return;

  layer = foundry_process_launcher_current_layer (self);

  copy = g_strdupv ((char **)args);
  g_array_append_vals (layer->argv, copy, g_strv_length (copy));
  g_free (copy);
}

gboolean
foundry_process_launcher_append_args_parsed (FoundryProcessLauncher  *self,
                                             const char              *args,
                                             GError                 **error)
{
  FoundryProcessLauncherLayer *layer;
  char **argv = NULL;
  int argc;

  g_return_val_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self), FALSE);
  g_return_val_if_fail (args != NULL, FALSE);

  layer = foundry_process_launcher_current_layer (self);

  if (!g_shell_parse_argv (args, &argc, &argv, error))
    return FALSE;

  g_array_append_vals (layer->argv, argv, argc);
  g_free (argv);

  return TRUE;
}

void
foundry_process_launcher_take_fd (FoundryProcessLauncher *self,
                                  int                     source_fd,
                                  int                     dest_fd)
{
  FoundryProcessLauncherLayer *layer;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_return_if_fail (source_fd >= -1);
  g_return_if_fail (dest_fd > -1);

  layer = foundry_process_launcher_current_layer (self);

  foundry_unix_fd_map_take (layer->unix_fd_map, source_fd, dest_fd);
}

const char *
foundry_process_launcher_getenv (FoundryProcessLauncher *self,
                                 const char             *key)
{
  FoundryProcessLauncherLayer *layer;
  gsize keylen;

  g_return_val_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self), NULL);
  g_return_val_if_fail (key != NULL, NULL);

  layer = foundry_process_launcher_current_layer (self);

  keylen = strlen (key);

  for (guint i = 0; i < layer->env->len; i++)
    {
      const char *envvar = g_array_index (layer->env, const char *, i);

      if (strncmp (key, envvar, keylen) == 0 && envvar[keylen] == '=')
        return &envvar[keylen+1];
    }

  return NULL;
}

void
foundry_process_launcher_setenv (FoundryProcessLauncher *self,
                                 const char             *key,
                                 const char             *value)
{
  FoundryProcessLauncherLayer *layer;
  char *element;
  gsize keylen;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_return_if_fail (key != NULL);

  if (value == NULL)
    {
      foundry_process_launcher_unsetenv (self, key);
      return;
    }

  layer = foundry_process_launcher_current_layer (self);

  keylen = strlen (key);
  element = g_strconcat (key, "=", value, NULL);

  g_array_append_val (layer->env, element);

  for (guint i = 0; i < layer->env->len-1; i++)
    {
      const char *envvar = g_array_index (layer->env, const char *, i);

      if (strncmp (key, envvar, keylen) == 0 && envvar[keylen] == '=')
        {
          g_array_remove_index_fast (layer->env, i);
          break;
        }
    }
}

void
foundry_process_launcher_unsetenv (FoundryProcessLauncher *self,
                                   const char             *key)
{
  FoundryProcessLauncherLayer *layer;
  gsize len;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_return_if_fail (key != NULL);

  layer = foundry_process_launcher_current_layer (self);

  len = strlen (key);

  for (guint i = 0; i < layer->env->len; i++)
    {
      const char *envvar = g_array_index (layer->env, const char *, i);

      if (strncmp (key, envvar, len) == 0 && envvar[len] == '=')
        {
          g_array_remove_index_fast (layer->env, i);
          return;
        }
    }
}

void
foundry_process_launcher_environ_to_argv (FoundryProcessLauncher *self)
{
  FoundryProcessLauncherLayer *layer;
  const char **copy;

  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  layer = foundry_process_launcher_current_layer (self);

  if (layer->env->len == 0)
    return;

  copy = (const char **)g_new0 (char *, layer->env->len + 2);
  copy[0] = "env";
  for (guint i = 0; i < layer->env->len; i++)
    copy[1+i] = g_array_index (layer->env, const char *, i);
  foundry_process_launcher_prepend_args (self, (const char * const *)copy);
  g_free (copy);

  g_array_set_size (layer->env, 0);
}

static gboolean
foundry_process_launcher_default_handler (FoundryProcessLauncher  *self,
                                          const char * const      *argv,
                                          const char * const      *env,
                                          const char              *cwd,
                                          FoundryUnixFDMap        *unix_fd_map,
                                          gpointer                 user_data,
                                          GError                 **error)
{
  FoundryProcessLauncherLayer *layer;

  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_assert (argv != NULL);
  g_assert (env != NULL);
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (unix_fd_map));

  layer = foundry_process_launcher_current_layer (self);

  if (cwd != NULL)
    {
      /* If the working directories do not match, we can't satisfy this and
       * need to error out.
       */
      if (layer->cwd != NULL && g_strcmp0 (cwd, layer->cwd) != 0)
        {
          g_set_error (error,
                       G_IO_ERROR,
                       G_IO_ERROR_INVALID_ARGUMENT,
                       "Cannot resolve differently requested cwd: %s and %s",
                       cwd, layer->cwd);
          return FALSE;
        }

      foundry_process_launcher_set_cwd (self, cwd);
    }

  /* Merge all the FDs unless there are collisions */
  if (!foundry_unix_fd_map_steal_from (layer->unix_fd_map, unix_fd_map, error))
    return FALSE;

  if (env[0] != NULL)
    {
      if (argv[0] == NULL)
        {
          foundry_process_launcher_add_environ (self, env);
        }
      else
        {
          foundry_process_launcher_append_argv (self, "env");
          foundry_process_launcher_append_args (self, env);
        }
    }

  if (argv[0] != NULL)
    foundry_process_launcher_append_args (self, argv);

  return TRUE;
}

static int
sort_strptr (gconstpointer a,
             gconstpointer b)
{
  const char * const *astr = a;
  const char * const *bstr = b;

  return g_strcmp0 (*astr, *bstr);
}

static gboolean
foundry_process_launcher_callback_layer (FoundryProcessLauncher       *self,
                                         FoundryProcessLauncherLayer  *layer,
                                         GError                      **error)
{
  FoundryProcessLauncherHandler handler;
  gpointer handler_data;
  gboolean ret;

  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (self));
  g_assert (layer != NULL);
  g_assert (layer != &self->root);

  handler = layer->handler ? layer->handler : foundry_process_launcher_default_handler;
  handler_data = layer->handler ? layer->handler_data : NULL;

  /* Sort environment variables first so that we have an easier time
   * finding them by eye in tooling which translates them.
   */
  g_array_sort (layer->env, sort_strptr);

  ret = handler (self,
                 (const char * const *)(gpointer)layer->argv->data,
                 (const char * const *)(gpointer)layer->env->data,
                 layer->cwd,
                 layer->unix_fd_map,
                 handler_data,
                 error);

  foundry_process_launcher_layer_free (layer);

  return ret;
}

static void
setup_tty (gpointer data)
{
  setsid ();
  setpgid (0, 0);

  if (isatty (STDIN_FILENO))
    {
      if (ioctl (STDIN_FILENO, TIOCSCTTY, 0) != 0)
        {
        }
    }
}

/**
 * foundry_process_launcher_spawn:
 * @self: a #FoundryProcessLauncher
 *
 * Spawns the run command.
 *
 * If there is a failure to build the command into a subprocess launcher,
 * then %NULL is returned and @error is set.
 *
 * If the subprocess fails to launch, then %NULL is returned and @error is set.
 *
 * Returns: (transfer full): an #GSubprocess if successful; otherwise %NULL
 *   and @error is set.
 */
GSubprocess *
foundry_process_launcher_spawn (FoundryProcessLauncher  *self,
                                GError                 **error)
{
  return foundry_process_launcher_spawn_with_flags (self, 0, error);
}

static gboolean
environ_parse (const char  *pair,
               char       **key,
               char       **value)
{
  const char *eq;

  g_assert (pair != NULL);

  if (key != NULL)
    *key = NULL;

  if (value != NULL)
    *value = NULL;

  if ((eq = strchr (pair, '=')))
    {
      if (key != NULL)
        *key = g_strndup (pair, eq - pair);

      if (value != NULL)
        *value = g_strdup (eq + 1);

      return TRUE;
    }

  return FALSE;
}

/**
 * foundry_process_launcher_spawn_with_flags:
 * @self: a #FoundryProcessLauncher
 *
 * Like foundry_process_launcher_spawn() but allows specifying the
 * flags for the GSubprocess which may override other settings.
 *
 * Returns: (transfer full): a #GSubprocess or %NULL upon error.
 */
GSubprocess *
foundry_process_launcher_spawn_with_flags (FoundryProcessLauncher  *self,
                                           GSubprocessFlags         flags,
                                           GError                 **error)
{
  g_autoptr(GSubprocessLauncher) launcher = NULL;
  const char * const *env;
  const char * const *argv;
  const char *cwd;
  guint length;

  g_return_val_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self), NULL);
  g_return_val_if_fail (self->ended == FALSE, NULL);

  self->ended = TRUE;

  while (self->layers.length > 1)
    {
      FoundryProcessLauncherLayer *layer = foundry_process_launcher_current_layer (self);

      g_queue_unlink (&self->layers, &layer->qlink);

      if (!foundry_process_launcher_callback_layer (self, layer, error))
        return FALSE;
    }

  argv = foundry_process_launcher_get_argv (self);
  env = foundry_process_launcher_get_environ (self);
  cwd = foundry_process_launcher_get_cwd (self);

  launcher = g_subprocess_launcher_new (0);

  FOUNDRY_TRACE_MSG ("Spawning a new process");

  if (env != NULL)
    {
      for (guint i = 0; env[i]; i++)
        {
          g_autofree char *key = NULL;
          g_autofree char *value = NULL;

          if (environ_parse (env[i], &key, &value))
            {
#ifdef FOUNDRY_ENABLE_TRACE
              g_autofree char *key_esc = g_strescape (key, NULL);
              g_autofree char *value_esc = g_strescape (value, NULL);

              FOUNDRY_TRACE_MSG ("Environment[%s] = %s", key_esc, value_esc);
#endif

              g_subprocess_launcher_setenv (launcher, key, value, TRUE);
            }
        }
    }

  FOUNDRY_TRACE_MSG ("Directory = %s", cwd);

  g_subprocess_launcher_set_cwd (launcher, cwd);

  length = foundry_unix_fd_map_get_length (self->root.unix_fd_map);

  for (guint i = 0; i < length; i++)
    {
      int source_fd;
      int dest_fd;

      source_fd = foundry_unix_fd_map_steal (self->root.unix_fd_map, i, &dest_fd);

      if (dest_fd == STDOUT_FILENO && source_fd == -1 && (flags & G_SUBPROCESS_FLAGS_STDOUT_PIPE) == 0)
        flags |= G_SUBPROCESS_FLAGS_STDOUT_SILENCE;

      if (dest_fd == STDERR_FILENO && source_fd == -1 && (flags & G_SUBPROCESS_FLAGS_STDERR_PIPE) == 0)
        flags |= G_SUBPROCESS_FLAGS_STDERR_SILENCE;

      if (source_fd != -1 && dest_fd != -1)
        {
          if (dest_fd == STDIN_FILENO)
            g_subprocess_launcher_take_stdin_fd (launcher, source_fd);
          else if (dest_fd == STDOUT_FILENO)
            g_subprocess_launcher_take_stdout_fd (launcher, source_fd);
          else if (dest_fd == STDERR_FILENO)
            g_subprocess_launcher_take_stderr_fd (launcher, source_fd);
          else
            g_subprocess_launcher_take_fd (launcher, source_fd, dest_fd);
        }
    }

  g_subprocess_launcher_set_flags (launcher, flags);

  if (self->setup_tty)
    g_subprocess_launcher_set_child_setup (launcher, setup_tty, NULL, NULL);

#ifdef FOUNDRY_ENABLE_TRACE
  for (guint i = 0; argv[i]; i++)
    {
      g_autofree char *arg_esc = g_strescape (argv[i], NULL);
      FOUNDRY_TRACE_MSG ("Argument[%d] = %s", i, arg_esc);
    }
#endif

  return g_subprocess_launcher_spawnv (launcher, argv, error);
}

/**
 * foundry_process_launcher_merge_unix_fd_map:
 * @self: a #FoundryProcessLauncher
 * @unix_fd_map: a #FoundryUnixFDMap
 * @error: a #GError, or %NULL
 *
 * Merges the #FoundryUnixFDMap into the current layer.
 *
 * If there are collisions in destination FDs, then that may cause an
 * error and %FALSE is returned.
 *
 * @unix_fd_map will have the FDs stolen using foundry_unix_fd_map_steal_from()
 * which means that if successful, @unix_fd_map will not have any open
 * file-descriptors after calling this function.
 *
 * Returns: %TRUE if successful; otherwise %FALSE and @error is set.
 */
gboolean
foundry_process_launcher_merge_unix_fd_map (FoundryProcessLauncher  *self,
                                            FoundryUnixFDMap        *unix_fd_map,
                                            GError                 **error)
{
  FoundryProcessLauncherLayer *layer;

  g_return_val_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self), FALSE);
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (unix_fd_map), FALSE);

  layer = foundry_process_launcher_current_layer (self);

  return foundry_unix_fd_map_steal_from (layer->unix_fd_map, unix_fd_map, error);
}

static int
pty_create_producer (int      consumer_fd,
                     gboolean blocking)
{
  g_autofd int ret = -1;
  int extra = blocking ? 0 : O_NONBLOCK;
#if defined(HAVE_PTSNAME_R) || defined(__FreeBSD__)
  char name[256];
#else
  const char *name;
#endif

  g_assert (consumer_fd != -1);

  if (grantpt (consumer_fd) != 0)
    return -1;

  if (unlockpt (consumer_fd) != 0)
    return -1;

#ifdef HAVE_PTSNAME_R
  if (ptsname_r (consumer_fd, name, sizeof name - 1) != 0)
    return -1;
  name[sizeof name - 1] = '\0';
#elif defined(__FreeBSD__)
  if (fdevname_r (consumer_fd, name + 5, sizeof name - 6) == NULL)
    return -1;
  memcpy (name, "/dev/", 5);
  name[sizeof name - 1] = '\0';
#else
  if (NULL == (name = ptsname (consumer_fd)))
    return -1;
#endif

  ret = open (name, O_NOCTTY | O_RDWR | O_CLOEXEC | extra);

  if (ret == -1 && errno == EINVAL)
    {
      gint flags;

      ret = open (name, O_NOCTTY | O_RDWR | O_CLOEXEC);
      if (ret == -1 && errno == EINVAL)
        ret = open (name, O_NOCTTY | O_RDWR);

      if (ret == -1)
        return -1;

      /* Add FD_CLOEXEC if O_CLOEXEC failed */
      flags = fcntl (ret, F_GETFD, 0);
      if ((flags & FD_CLOEXEC) == 0)
        {
          if (fcntl (ret, F_SETFD, flags | FD_CLOEXEC) < 0)
            return -1;
        }

      if (!blocking)
        {
          if (!g_unix_set_fd_nonblocking (ret, TRUE, NULL))
            return -1;
        }
    }

  return g_steal_fd (&ret);
}

/**
 * foundry_process_launcher_set_pty_fd:
 * @self: an #FoundryProcessLauncher
 * @consumer_fd: the FD of the PTY consumer
 *
 * Sets up a PTY for the run context that will communicate with the
 * consumer. The consumer is the generally the widget that is rendering
 * the PTY contents and the producer is the FD that is connected to the
 * subprocess.
 */
void
foundry_process_launcher_set_pty_fd (FoundryProcessLauncher *self,
                                     int                     consumer_fd)
{
  int stdin_fd = -1;
  int stdout_fd = -1;
  int stderr_fd = -1;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self));

  if (consumer_fd < 0)
    return;

  if (-1 == (stdin_fd = pty_create_producer (consumer_fd, TRUE)))
    {
      int errsv = errno;
      g_critical ("Failed to create PTY device: %s", g_strerror (errsv));
      return;
    }

  if (-1 == (stdout_fd = dup (stdin_fd)))
    {
      int errsv = errno;
      g_critical ("Failed to dup stdout FD: %s", g_strerror (errsv));
    }

  if (-1 == (stderr_fd = dup (stdin_fd)))
    {
      int errsv = errno;
      g_critical ("Failed to dup stderr FD: %s", g_strerror (errsv));
    }

  g_assert (stdin_fd > -1);
  g_assert (stdout_fd > -1);
  g_assert (stderr_fd > -1);

  foundry_process_launcher_take_fd (self, stdin_fd, STDIN_FILENO);
  foundry_process_launcher_take_fd (self, stdout_fd, STDOUT_FILENO);
  foundry_process_launcher_take_fd (self, stderr_fd, STDERR_FILENO);
}

/**
 * foundry_process_launcher_create_stdio_stream:
 * @self: a #FoundryProcessLauncher
 * @error: a location for a #GError
 *
 * Creates a stream to communicate with the subprocess using stdin/stdout.
 *
 * The stream is created using UNIX pipes which are attached to the
 * stdin/stdout of the child process.
 *
 * Returns: (transfer full): a #GIOStream if successful; otherwise
 *   %NULL and @error is set.
 */
GIOStream *
foundry_process_launcher_create_stdio_stream (FoundryProcessLauncher  *self,
                                              GError                 **error)
{
  FoundryProcessLauncherLayer *layer;

  g_return_val_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (self), NULL);

  layer = foundry_process_launcher_current_layer (self);

  return foundry_unix_fd_map_create_stream (layer->unix_fd_map,
                                            STDIN_FILENO,
                                            STDOUT_FILENO,
                                            error);
}

int
foundry_pty_create_producer (int        pty_consumer_fd,
                             gboolean   blocking,
                             GError   **error)
{
  int fd = pty_create_producer (pty_consumer_fd, blocking);

  if (fd == -1)
    {
      int errsv = errno;
      g_set_error_literal (error,
                           G_IO_ERROR,
                           g_io_error_from_errno (errsv),
                           g_strerror (errsv));
    }

  return fd;
}
