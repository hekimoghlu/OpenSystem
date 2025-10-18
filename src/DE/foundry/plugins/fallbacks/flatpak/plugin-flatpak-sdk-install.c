/* plugin-flatpak-sdk-install.c
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

#include <glib/gi18n-lib.h>

#include "plugin-flatpak.h"
#include "plugin-flatpak-sdk-private.h"

typedef struct _Install
{
  FoundryContext      *context;
  FoundryOperation    *operation;
  FlatpakInstallation *installation;
  FlatpakRef          *ref;
  FlatpakTransaction  *transaction;
  DexPromise          *promise;
  DexCancellable      *cancellable;
  GCancellable        *gcancellable;
  guint                do_update : 1;
} Install;

static void
install_finalize (gpointer data)
{
  Install *install = data;

  g_clear_object (&install->context);
  g_clear_object (&install->operation);
  g_clear_object (&install->installation);
  g_clear_object (&install->ref);
  g_clear_object (&install->transaction);
  g_clear_object (&install->gcancellable);
  dex_clear (&install->promise);
  dex_clear (&install->cancellable);
}

static void
install_unref (Install *install)
{
  return g_atomic_rc_box_release_full (install, install_finalize);
}

static Install *
install_ref (Install *install)
{
  return g_atomic_rc_box_acquire (install);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (Install, install_unref)

static void
handle_progress_changed (FlatpakTransactionProgress *progress,
                         FoundryOperation           *operation)
{
  g_autofree char *status = NULL;
  double fraction;

  g_assert (FLATPAK_IS_TRANSACTION_PROGRESS (progress));
  g_assert (FOUNDRY_IS_OPERATION (operation));

  status = flatpak_transaction_progress_get_status (progress);
  fraction = flatpak_transaction_progress_get_progress (progress) / 100.;

  foundry_operation_set_subtitle (operation, status);
  foundry_operation_set_progress (operation, fraction);
}

static void
handle_new_operation (FlatpakTransaction          *object,
                      FlatpakTransactionOperation *operation,
                      FlatpakTransactionProgress  *progress,
                      FoundryOperation            *foundry_op)
{
  g_signal_connect_object (progress,
                           "changed",
                           G_CALLBACK (handle_progress_changed),
                           foundry_op,
                           0);

  handle_progress_changed (progress, foundry_op);
}

static gpointer
plugin_flatpak_sdk_install_thread (gpointer user_data)
{
  g_autoptr(Install) install = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (install != NULL);
  g_assert (FOUNDRY_IS_OPERATION (install->operation));
  g_assert (FLATPAK_IS_INSTALLATION (install->installation));
  g_assert (FLATPAK_IS_REF (install->ref));
  g_assert (FLATPAK_IS_TRANSACTION (install->transaction));
  g_assert (DEX_IS_PROMISE (install->promise));

  if (!flatpak_transaction_run (install->transaction,
                                install->gcancellable,
                                &error))
    dex_promise_reject (install->promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (install->promise, TRUE);

  foundry_operation_complete (install->operation);

  return NULL;
}

static DexFuture *
plugin_flatpak_sdk_install_fiber (gpointer user_data)
{
  Install *install = user_data;
  g_autoptr(FlatpakTransaction) transaction = NULL;
  g_autoptr(FlatpakRemote) remote = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *ref_str = NULL;
  g_autofree char *title = NULL;

  g_assert (install != NULL);
  g_assert (FOUNDRY_IS_OPERATION (install->operation));
  g_assert (FLATPAK_IS_INSTALLATION (install->installation));
  g_assert (FLATPAK_IS_REF (install->ref));
  g_assert (DEX_IS_PROMISE (install->promise));
  g_assert (DEX_IS_CANCELLABLE (install->cancellable));

  ref_str = flatpak_ref_format_ref (install->ref);
  title = g_strdup_printf ("%s %s", _("Installing"), ref_str);

  foundry_operation_set_title (install->operation, title);

  if (!(transaction = flatpak_transaction_new_for_installation (install->installation, NULL, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(remote = plugin_flatpak_find_remote (install->context, install->installation, install->ref)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Failed to find remote for %s", ref_str);

  if (install->do_update)
    flatpak_transaction_add_update (transaction, ref_str, NULL, NULL, &error);
  else
    flatpak_transaction_add_install (transaction, flatpak_remote_get_name (remote), ref_str, NULL, &error);

  if (error != NULL)
    return dex_future_new_for_error (g_steal_pointer (&error));

  flatpak_transaction_set_no_interaction (transaction, TRUE);

  if (flatpak_transaction_is_empty (transaction))
    {
      foundry_operation_complete (install->operation);
      return dex_future_new_true ();
    }

  g_signal_connect_object (transaction,
                           "new-operation",
                           G_CALLBACK (handle_new_operation),
                           install->operation,
                           0);

  /* Run transaction in a thread so we don't hold up the
   * thread pool with the long running action.
   */
  install->transaction = g_object_ref (transaction);
  g_thread_unref (g_thread_new ("[foundry-flatpak-install]",
                                plugin_flatpak_sdk_install_thread,
                                install_ref (install)));

  /* Wait for completion. Force cancellation of GCancellable so that if
   * our fiber is cancelled the thread cancels too.
   */
  if (!dex_await (dex_future_first (dex_ref (DEX_FUTURE (install->promise)),
                                    dex_ref (DEX_FUTURE (install->cancellable)),
                                    NULL),
                  NULL))
    g_cancellable_cancel (install->gcancellable);

  return dex_future_new_true ();
}

DexFuture *
plugin_flatpak_sdk_install (FoundrySdk       *sdk,
                            FoundryOperation *operation,
                            DexCancellable   *cancellable)
{
  PluginFlatpakSdk *self = (PluginFlatpakSdk *)sdk;
  Install *install;

  g_assert (PLUGIN_IS_FLATPAK_SDK (self));
  g_assert (FOUNDRY_IS_OPERATION (operation));

  install = g_atomic_rc_box_new0 (Install);
  install->context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (sdk));
  install->operation = g_object_ref (operation);
  install->installation = g_object_ref (self->installation);
  install->ref = g_object_ref (self->ref);
  install->promise = dex_promise_new_cancellable ();
  install->do_update = foundry_sdk_get_installed (sdk);
  install->cancellable = cancellable ? dex_ref (cancellable) : dex_cancellable_new ();
  install->gcancellable = g_cancellable_new ();

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              plugin_flatpak_sdk_install_fiber,
                              install,
                              (GDestroyNotify) install_unref);
}

DexFuture *
plugin_flatpak_ref_install (FoundryContext      *context,
                            FlatpakInstallation *installation,
                            FlatpakRef          *ref,
                            FoundryOperation    *operation,
                            gboolean             is_installed,
                            DexCancellable      *cancellable)
{
  Install *install;

  g_assert (FOUNDRY_IS_CONTEXT (context));
  g_assert (FLATPAK_IS_INSTALLATION (installation));
  g_assert (FLATPAK_IS_REF (ref));
  g_assert (FOUNDRY_IS_OPERATION (operation));

  install = g_atomic_rc_box_new0 (Install);
  install->context = g_object_ref (context);
  install->operation = g_object_ref (operation);
  install->installation = g_object_ref (installation);
  install->ref = g_object_ref (ref);
  install->promise = dex_promise_new_cancellable ();
  install->do_update = !!is_installed;
  install->cancellable = cancellable ? dex_ref (cancellable) : dex_cancellable_new ();
  install->gcancellable = g_cancellable_new ();

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              plugin_flatpak_sdk_install_fiber,
                              install,
                              (GDestroyNotify) install_unref);
}
