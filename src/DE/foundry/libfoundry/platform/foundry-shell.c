/* foundry-shell.c
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

#include <gio/gio.h>

#include "foundry-debug.h"
#include "foundry-shell-private.h"
#include "foundry-util-private.h"

static const char *user_shell = "/bin/sh";
static const char *user_default_path = SAFE_PATH;
static DexPromise *shell_init;

gboolean
foundry_shell_supports_dash_c (const char *shell)
{
  if (shell == NULL)
    return FALSE;

  return strcmp (shell, "bash") == 0 || g_str_has_suffix (shell, "/bash") ||
#if 0
         /* Fish does apparently support -l and -c in testing, but it is causing
          * issues with users, so we will disable it for now so that we fallback
          * to using `sh -l -c ''` instead.
          */
         strcmp (shell, "fish") == 0 || g_str_has_suffix (shell, "/fish") ||
#endif
         strcmp (shell, "zsh") == 0 || g_str_has_suffix (shell, "/zsh") ||
         strcmp (shell, "dash") == 0 || g_str_has_suffix (shell, "/dash") ||
         strcmp (shell, "tcsh") == 0 || g_str_has_suffix (shell, "/tcsh") ||
         strcmp (shell, "sh") == 0 || g_str_has_suffix (shell, "/sh");
}

/**
 * foundry_shell_supports_dash_login:
 * @shell: the name of the shell, such as `sh` or `/bin/sh`
 *
 * Checks if the shell is known to support login semantics. Originally,
 * this meant `--login`, but now is meant to mean `-l` as more shells
 * support `-l` than `--login` (notably dash).
 *
 * Returns: %TRUE if @shell likely supports `-l`.
 */
gboolean
foundry_shell_supports_dash_login (const char *shell)
{
  if (shell == NULL)
    return FALSE;

  return strcmp (shell, "bash") == 0 || g_str_has_suffix (shell, "/bash") ||
#if 0
         strcmp (shell, "fish") == 0 || g_str_has_suffix (shell, "/fish") ||
#endif
         strcmp (shell, "zsh") == 0 || g_str_has_suffix (shell, "/zsh") ||
         strcmp (shell, "dash") == 0 || g_str_has_suffix (shell, "/dash") ||
#if 0
         /* tcsh supports -l and -c but not combined! To do that, you'd have
          * to instead launch the login shell like `-tcsh -c 'command'`, which
          * is possible, but we lack the abstractions for that currently.
          */
         strcmp (shell, "tcsh") == 0 || g_str_has_suffix (shell, "/tcsh") ||
#endif
         strcmp (shell, "sh") == 0 || g_str_has_suffix (shell, "/sh");
}

static void
foundry_guess_shell_communicate_cb (GObject      *object,
                                    GAsyncResult *result,
                                    gpointer      user_data)
{
  GSubprocess *subprocess = (GSubprocess *)object;
  g_autoptr(GTask) task = user_data;
  g_autoptr(GError) error = NULL;
  g_autofree char *stdout_buf = NULL;
  const char *key;

  FOUNDRY_ENTRY;

  g_assert (G_IS_SUBPROCESS (subprocess));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (G_IS_TASK (task));

  key = g_task_get_task_data (task);

  if (!g_subprocess_communicate_utf8_finish (subprocess, result, &stdout_buf, NULL, &error))
    {
      g_task_return_error (task, g_steal_pointer (&error));
      FOUNDRY_EXIT;
    }

  if (stdout_buf != NULL)
    g_strstrip (stdout_buf);

  g_debug ("Guessed %s as \"%s\"", key, stdout_buf);

  if (foundry_str_equal0 (key, "SHELL"))
    {
      if (stdout_buf[0] == '/')
        user_shell = g_steal_pointer (&stdout_buf);
    }
  else if (foundry_str_equal0 (key, "PATH"))
    {
      if (!foundry_str_empty0 (stdout_buf))
        user_default_path = g_steal_pointer (&stdout_buf);
    }
  else
    {
      g_critical ("Unknown key %s", key);
    }

  g_task_return_boolean (task, TRUE);

  FOUNDRY_EXIT;
}

static void
_foundry_guess_shell (GCancellable        *cancellable,
                      GAsyncReadyCallback  callback,
                      gpointer             user_data)
{
  g_autoptr(GTask) task = NULL;
  g_autoptr(GSubprocessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GPtrArray) args = NULL;
  g_autofree char *command = NULL;
  g_autoptr(GError) error = NULL;
  g_auto(GStrv) argv = NULL;

  FOUNDRY_ENTRY;

  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));

  task = g_task_new (NULL, cancellable, callback, user_data);
  g_task_set_task_data (task, g_strdup ("SHELL"), g_free);

#ifdef __APPLE__
  command = g_strdup_printf ("sh -c 'dscacheutil -q user -a name %s | grep ^shell: | cut -f 2 -d \" \"'",
                             g_get_user_name ());
#else
  command = g_strdup_printf ("sh -c 'getent passwd %s | head -n1 | cut -f 7 -d :'",
                             g_get_user_name ());
#endif

  if (!g_shell_parse_argv (command, NULL, &argv, &error))
    {
      g_task_return_error (task, g_steal_pointer (&error));
      FOUNDRY_EXIT;
    }

  /*
   * We don't use the runtime shell here, because we want to know
   * what the host thinks the user shell should be.
   */
  launcher = g_subprocess_launcher_new (G_SUBPROCESS_FLAGS_STDOUT_PIPE);
  g_subprocess_launcher_set_cwd (launcher, g_get_home_dir ());

  args = g_ptr_array_new ();

  if (_foundry_in_container ())
    {
      g_ptr_array_add (args, (char *)"flatpak-spawn");
      g_ptr_array_add (args, (char *)"--host");
      g_ptr_array_add (args, (char *)"--watch-bus");
    }

  for (guint i = 0; argv[i]; i++)
    g_ptr_array_add (args, argv[i]);
  g_ptr_array_add (args, NULL);

  if (!(subprocess = g_subprocess_launcher_spawnv (launcher, (const char * const *)args->pdata, &error)))
    g_task_return_error (task, g_steal_pointer (&error));
  else
    g_subprocess_communicate_utf8_async (subprocess,
                                         NULL,
                                         cancellable,
                                         foundry_guess_shell_communicate_cb,
                                         g_steal_pointer (&task));

  FOUNDRY_EXIT;
}

static void
_foundry_guess_user_path (GCancellable        *cancellable,
                          GAsyncReadyCallback  callback,
                          gpointer             user_data)
{
  g_autoptr(GSubprocessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GPtrArray) argv = NULL;
  g_autoptr(GTask) task = NULL;
  g_autoptr(GError) error = NULL;

  FOUNDRY_ENTRY;

  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));

  argv = g_ptr_array_new ();

  task = g_task_new (NULL, cancellable, callback, user_data);
  g_task_set_task_data (task, g_strdup ("PATH"), g_free);

  /* This works by running 'echo $PATH' on the host, preferably
   * through the user $SHELL we discovered.
   */
  launcher = g_subprocess_launcher_new (G_SUBPROCESS_FLAGS_STDOUT_PIPE);
  g_subprocess_launcher_set_cwd (launcher, g_get_home_dir ());

  if (_foundry_in_container ())
    {
      g_ptr_array_add (argv, (char *)"flatpak-spawn");
      g_ptr_array_add (argv, (char *)"--host");
      g_ptr_array_add (argv, (char *)"--watch-bus");
    }

  if (foundry_shell_supports_dash_c (user_shell))
    {
      g_ptr_array_add (argv, (char *)user_shell);
      if (foundry_shell_supports_dash_login (user_shell))
        g_ptr_array_add (argv, (char *)"-l");
      g_ptr_array_add (argv, (char *)"-c");
      g_ptr_array_add (argv, (char *)"echo $PATH");
    }
  else
    {
      g_ptr_array_add (argv, (char *)"/bin/sh");
      g_ptr_array_add (argv, (char *)"-l");
      g_ptr_array_add (argv, (char *)"-c");
      g_ptr_array_add (argv, (char *)"echo $PATH");
    }

  g_ptr_array_add (argv, NULL);

  if (!(subprocess = g_subprocess_launcher_spawnv (launcher, (const char * const *)argv->pdata, &error)))
    g_task_return_error (task, g_steal_pointer (&error));
  else
    g_subprocess_communicate_utf8_async (subprocess,
                                         NULL,
                                         NULL,
                                         foundry_guess_shell_communicate_cb,
                                         g_steal_pointer (&task));

  FOUNDRY_EXIT;
}

/**
 * foundry_shell_get_default:
 *
 * Gets the user preferred shell on the host.
 *
 * If the background shell discovery has not yet finished due to
 * slow or misconfigured getent on the host, this will provide a
 * sensible fallback.
 *
 * Returns: (not nullable): a shell such as "/bin/sh"
 */
const char *
foundry_shell_get_default (void)
{
  return user_shell;
}

/**
 * foundry_shell_get_default_path:
 *
 * Gets the default `$PATH` on the system for the user on the host.
 *
 * This value is sniffed during startup and will default to `SAFE_PATH`
 * configured when building Builder until that value has been discovered.
 *
 * Returns: (not nullable): a string such as "/bin:/usr/bin"
 */
const char *
foundry_shell_get_default_path (void)
{
  return user_default_path;
}

static void
foundry_shell_init_guess_path_cb (GObject      *object,
                                  GAsyncResult *result,
                                  gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  FOUNDRY_ENTRY;

  g_assert (object == NULL);
  g_assert (G_IS_TASK (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!g_task_propagate_boolean (G_TASK (result), &error))
    g_message ("Failed to guess user $PATH using $SHELL %s: %s",
               user_shell, error->message);

  if (error != NULL)
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);

  FOUNDRY_EXIT;
}

static void
foundry_shell_init_guess_shell_cb (GObject      *object,
                                   GAsyncResult *result,
                                   gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  FOUNDRY_ENTRY;

  g_assert (object == NULL);
  g_assert (G_IS_TASK (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!g_task_propagate_boolean (G_TASK (result), &error))
    g_message ("Failed to guess user $SHELL: %s", error->message);

  _foundry_guess_user_path (NULL,
                            foundry_shell_init_guess_path_cb,
                            g_steal_pointer (&promise));

  FOUNDRY_EXIT;
}

DexFuture *
_foundry_shell_init (void)
{
  FOUNDRY_ENTRY;

  if (g_once_init_enter (&shell_init))
    {
      g_once_init_leave (&shell_init, dex_promise_new ());
      _foundry_guess_shell (NULL,
                            foundry_shell_init_guess_shell_cb,
                            dex_ref (shell_init));
    }

  FOUNDRY_RETURN (dex_ref (DEX_FUTURE (shell_init)));
}
