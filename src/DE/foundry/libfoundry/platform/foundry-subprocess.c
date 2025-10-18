/* foundry-subprocess.c
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

#include "foundry-subprocess.h"
#include "foundry-util-private.h"

static void
foundry_subprocess_communicate_utf8_cb (GObject      *object,
                                        GAsyncResult *result,
                                        gpointer      user_data)
{
  GSubprocess *subprocess = (GSubprocess *)object;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;
  g_autofree char *stdout_buf = NULL;

  g_assert (G_IS_SUBPROCESS (subprocess));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!g_subprocess_communicate_utf8_finish (subprocess, result, &stdout_buf, NULL, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_string (promise, g_steal_pointer (&stdout_buf));
}

/**
 * foundry_subprocess_communicate_utf8:
 * @subprocess: a #FoundrySubprocess
 * @stdin_buf: the standard input buffer
 *
 * Like g_subprocess_communicate_utf8() but only supports stdout and is
 * returned as a future to a string.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a string
 */
DexFuture *
foundry_subprocess_communicate_utf8 (GSubprocess *subprocess,
                                     const char  *stdin_buf)
{
  DexPromise *promise;

  dex_return_error_if_fail (G_IS_SUBPROCESS (subprocess));

  promise = dex_promise_new_cancellable ();
  g_subprocess_communicate_utf8_async (subprocess,
                                       stdin_buf,
                                       dex_promise_get_cancellable (promise),
                                       foundry_subprocess_communicate_utf8_cb,
                                       dex_ref (promise));
  return DEX_FUTURE (promise);
}

static void
foundry_subprocess_communicate_cb (GObject      *object,
                                   GAsyncResult *result,
                                   gpointer      user_data)
{
  GSubprocess *subprocess = (GSubprocess *)object;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GBytes) stdout_bytes = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (G_IS_SUBPROCESS (subprocess));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!g_subprocess_communicate_finish (subprocess, result, &stdout_bytes, NULL, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    foundry_promise_resolve_bytes (promise, g_steal_pointer (&stdout_bytes));
}

/**
 * foundry_subprocess_communicate:
 * @subprocess: a #FoundrySubprocess
 * @stdin_bytes: (nullable): the standard input buffer
 *
 * Like g_subprocess_communicate() but only supports stdout and is
 * returned as a future to #GBytes.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a string
 */
DexFuture *
foundry_subprocess_communicate (GSubprocess *subprocess,
                                GBytes      *stdin_bytes)
{
  DexPromise *promise;

  dex_return_error_if_fail (G_IS_SUBPROCESS (subprocess));

  promise = dex_promise_new_cancellable ();
  g_subprocess_communicate_async (subprocess,
                                  stdin_bytes,
                                  dex_promise_get_cancellable (promise),
                                  foundry_subprocess_communicate_cb,
                                  dex_ref (promise));
  return DEX_FUTURE (promise);
}

typedef struct _WaitCheck
{
  GSubprocess    *subprocess;
  DexCancellable *cancellable;
} WaitCheck;

static void
wait_check_free (WaitCheck *state)
{
  g_clear_object (&state->subprocess);
  dex_clear (&state->cancellable);
  g_free (state);
}

static DexFuture *
foundry_subprocess_wait_check_fiber (gpointer data)
{
  WaitCheck *state = data;
  g_autoptr(GError) error = NULL;

  g_assert (state != NULL);
  g_assert (G_IS_SUBPROCESS (state->subprocess));
  g_assert (DEX_IS_CANCELLABLE (state->cancellable));

  dex_await (dex_future_first (dex_ref (state->cancellable),
                               dex_subprocess_wait_check (state->subprocess),
                               NULL),
             &error);

  if (g_error_matches (error, G_IO_ERROR, G_IO_ERROR_CANCELLED))
    g_subprocess_force_exit (state->subprocess);

  if (error != NULL)
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

/**
 * foundry_subprocess_wait_check:
 * @subprocess: a [class@Gio.Subprocess]
 * @cancellable: a [class@Dex.Cancellable]
 *
 * If @cancellable is cancelled, then @subprocess will be force exited.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a boolean
 */
DexFuture *
foundry_subprocess_wait_check (GSubprocess    *subprocess,
                               DexCancellable *cancellable)
{
  WaitCheck *state;

  g_return_val_if_fail (G_IS_SUBPROCESS (subprocess), NULL);
  g_return_val_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable), NULL);

  state = g_new0 (WaitCheck, 1);
  state->subprocess = g_object_ref (subprocess);
  state->cancellable = cancellable ? dex_ref (cancellable) : dex_cancellable_new ();

  return dex_scheduler_spawn (NULL, 0,
                              foundry_subprocess_wait_check_fiber,
                              state,
                              (GDestroyNotify) wait_check_free);
}
