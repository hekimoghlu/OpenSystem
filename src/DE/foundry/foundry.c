/* foundry.c
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

#include <locale.h>

#include <glib/gi18n.h>

#include <foundry.h>

#include "foundry-context-private.h"
#include "foundry-init-private.h"

static int exit_code;
static GMainLoop *main_loop;

static DexFuture *
shutdown_cb (DexFuture *completed,
             gpointer   user_data)
{
  exit_code = dex_await_int (dex_ref (completed), NULL);
  return _foundry_context_shutdown_all ();
}

static DexFuture *
run_cb (DexFuture *completed,
        gpointer   user_data)
{
  g_main_loop_quit (main_loop);
  g_clear_pointer (&main_loop, g_main_loop_unref);
  return NULL;
}

static char **
copy_argv (int    argc,
           char **argv)
{
  char **args;

  args = g_new (char *, argc + 1);
  for (int i = 0; i < argc; i++)
    args[i] = g_strdup (argv[i]);
  args[argc] = NULL;

  return args;
}

int
main (int   argc,
      char *argv[])
{
  g_autoptr(FoundryCommandLine) command_line = NULL;
  g_autoptr(DexFuture) future = NULL;
  g_auto(GStrv) args = NULL;

  setlocale (LC_ALL, "");
  bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);
  bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
  textdomain (GETTEXT_PACKAGE);

  main_loop = g_main_loop_new (NULL, FALSE);
  exit_code = EXIT_FAILURE;

  g_unsetenv ("G_MESSAGES_DEBUG");

  dex_future_disown (foundry_init ());

  args = copy_argv (argc, argv);
  command_line = foundry_command_line_new ();

  future = foundry_command_line_run (command_line, (const char * const *)args);
  future = dex_future_finally (future, shutdown_cb, NULL, NULL);
  future = dex_future_finally (future, run_cb, NULL, NULL);

  if (main_loop != NULL)
    g_main_loop_run (main_loop);

  return exit_code;
}
