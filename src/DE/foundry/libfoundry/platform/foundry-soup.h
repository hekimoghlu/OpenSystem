/* foundry-soup.h
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

#include <foundry.h>
#include <libsoup/soup.h>

G_BEGIN_DECLS

#ifndef __GI_SCANNER__

static void
foundry_soup_session_send_and_read_cb (GObject      *object,
                                       GAsyncResult *result,
                                       gpointer      user_data)
{
  SoupSession *session = (SoupSession *)object;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;
  g_autoptr(GBytes) bytes = NULL;

  g_assert (SOUP_IS_SESSION (session));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!(bytes = soup_session_send_and_read_finish (session, result, &error)))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boxed (promise, G_TYPE_BYTES, g_steal_pointer (&bytes));
}

static inline DexFuture *
foundry_soup_session_send_and_read (SoupSession *session,
                                    SoupMessage *message)
{
  DexPromise *promise = dex_promise_new_cancellable ();
  soup_session_send_and_read_async (session, message, G_PRIORITY_DEFAULT,
                                    dex_promise_get_cancellable (promise),
                                    foundry_soup_session_send_and_read_cb,
                                    dex_ref (promise));
  return DEX_FUTURE (promise);
}

#endif

G_END_DECLS
