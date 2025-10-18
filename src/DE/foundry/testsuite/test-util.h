/* test-util.h
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

#pragma once

#include <libdex.h>
#include <foundry.h>

G_BEGIN_DECLS

static inline DexFuture *
test_from_fiber_wrapper (gpointer data)
{
  GCallback callback = data;
  callback ();
  return dex_future_new_true ();
}

static inline DexFuture *
test_util_exit_main (DexFuture *completed,
                     gpointer   user_data)
{
  GMainLoop **main_loop = user_data;
  g_main_loop_quit (*main_loop);
  g_clear_pointer (main_loop, g_main_loop_unref);
  return dex_future_new_true ();
}

static inline void
test_from_fiber (GCallback callback)
{
  g_autoptr(GMainLoop) main_loop = g_main_loop_new (NULL, FALSE);
  DexFuture *future;

  future = dex_scheduler_spawn (NULL, 0,
                                test_from_fiber_wrapper,
                                callback,
                                NULL);
  future = dex_future_finally (future,
                               test_util_exit_main,
                               &main_loop,
                               NULL);
  dex_future_disown (future);

  if (main_loop != NULL)
    g_main_loop_run (main_loop);
}

static inline void
rm_rf (const char *path)
{
  g_autoptr(FoundryDirectoryReaper) reaper = NULL;
  g_autoptr(GFile) directory = NULL;

  directory = g_file_new_for_path (path);
  reaper = foundry_directory_reaper_new ();
  foundry_directory_reaper_add_directory (reaper, directory, 0);
  foundry_directory_reaper_add_file (reaper, directory, 0);

  dex_await (foundry_directory_reaper_execute (reaper), NULL);
}

G_END_DECLS
