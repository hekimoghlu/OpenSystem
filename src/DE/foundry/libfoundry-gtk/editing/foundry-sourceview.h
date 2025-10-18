/* foundry-sourceview.h
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
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

#include <gtksourceview/gtksource.h>
#include <libdex.h>

G_BEGIN_DECLS

static inline void
gtk_source_file_loader_load_cb (GObject      *object,
                                GAsyncResult *result,
                                gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  if (!gtk_source_file_loader_load_finish (GTK_SOURCE_FILE_LOADER (object), result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);
}

static inline DexFuture *
gtk_source_file_loader_load (GtkSourceFileLoader *loader,
                             int                  io_priority,
                             FoundryOperation    *operation)

{
  DexPromise *promise;

  dex_return_error_if_fail (GTK_SOURCE_IS_FILE_LOADER (loader));

  promise = dex_promise_new_cancellable ();

  gtk_source_file_loader_load_async (loader,
                                     io_priority,
                                     dex_promise_get_cancellable (promise),
                                     operation ? foundry_operation_file_progress : NULL,
                                     operation ? g_object_ref (operation) : NULL,
                                     operation ? g_object_unref : NULL,
                                     gtk_source_file_loader_load_cb,
                                     dex_ref (promise));

  return DEX_FUTURE (promise);
}

static inline void
gtk_source_file_saver_save_cb (GObject      *object,
                               GAsyncResult *result,
                               gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  if (!gtk_source_file_saver_save_finish (GTK_SOURCE_FILE_SAVER (object), result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);
}

static inline DexFuture *
gtk_source_file_saver_save (GtkSourceFileSaver *saver,
                            int                 io_priority,
                            FoundryOperation   *operation)

{
  DexPromise *promise;

  dex_return_error_if_fail (GTK_SOURCE_IS_FILE_SAVER (saver));

  promise = dex_promise_new_cancellable ();

  gtk_source_file_saver_save_async (saver,
                                    io_priority,
                                    dex_promise_get_cancellable (promise),
                                    operation ? foundry_operation_file_progress : NULL,
                                    operation ? g_object_ref (operation) : NULL,
                                    operation ? g_object_unref : NULL,
                                    gtk_source_file_saver_save_cb,
                                    dex_ref (promise));

  return DEX_FUTURE (promise);
}

G_END_DECLS
