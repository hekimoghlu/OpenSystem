/* test-diagnostics.c
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

#include "config.h"

#include <foundry.h>

static const char *filename;
static FoundryOnTypeDiagnostics *diagnostics;

static void
items_changed_cb (GListModel *model,
                  guint       position,
                  guint       n_removed,
                  guint       n_added)
{
  g_printerr ("Position: %u, Removed: %u, Added: %u\n", position, n_removed, n_added);

  for (guint i = position; i < position + n_added; i++)
    {
      g_autoptr(FoundryDiagnostic) diag = g_list_model_get_item (model, i);
      g_autofree char *message = foundry_diagnostic_dup_message (diag);

      g_printerr ("%s\n", message);
    }
}

static DexFuture *
main_fiber (gpointer user_data)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryTextManager) text_manager = NULL;
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryOperation) operation = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) file = NULL;
  gboolean r;
  guint n_items;

  r = dex_await (foundry_init (), &error);
  g_assert_no_error (error);
  g_assert_true (r);

  cancellable = dex_cancellable_new ();

  context = dex_await_object (foundry_context_new_for_user (cancellable), &error);
  g_assert_no_error (error);
  g_assert_nonnull (context);
  g_assert_true (FOUNDRY_IS_CONTEXT (context));

  text_manager = foundry_context_dup_text_manager (context);
  file = g_file_new_for_path (filename);
  operation = foundry_operation_new ();

  document = dex_await_object (foundry_text_manager_load (text_manager, file, operation, NULL), &error);
  g_assert_no_error (error);
  g_assert_nonnull (document);
  g_assert_true (FOUNDRY_IS_TEXT_DOCUMENT (document));

  diagnostics = foundry_on_type_diagnostics_new (document);

  g_signal_connect (diagnostics,
                    "items-changed",
                    G_CALLBACK (items_changed_cb),
                    NULL);

  if ((n_items = g_list_model_get_n_items (G_LIST_MODEL (diagnostics))))
    items_changed_cb (G_LIST_MODEL (diagnostics), 0, 0, n_items);

  return dex_future_new_true ();
}

int
main (int   argc,
      char *argv[])
{
  g_autoptr(GMainLoop) main_loop = g_main_loop_new (NULL, FALSE);

  if (g_strv_length ((char **)argv) != 2)
    g_error ("usage: %s FILENAME", argv[0]);

  filename = argv[1];

  dex_init ();

  g_print ("Waiting 30 seconds for changes...\n");
  g_timeout_add_seconds_once (30, (GSourceOnceFunc)g_main_loop_quit, main_loop);

  dex_future_disown (dex_scheduler_spawn (NULL, 0, main_fiber, NULL, NULL));

  g_main_loop_run (main_loop);

  return 0;
}
