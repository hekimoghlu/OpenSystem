/* plugin-deviced-dex.h
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

#include <libdeviced.h>
#include <libdex.h>

G_BEGIN_DECLS

static inline void
devd_client_connect_cb (GObject      *object,
                        GAsyncResult *result,
                        gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  if (!devd_client_connect_finish (DEVD_CLIENT (object), result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_object (promise, g_object_ref (object));
}

static inline DexFuture *
devd_client_connect (DevdClient *client)
{
  DexPromise *promise = dex_promise_new_cancellable ();
  devd_client_connect_async (client,
                             dex_promise_get_cancellable (promise),
                             devd_client_connect_cb,
                             dex_ref (promise));
  return DEX_FUTURE (promise);
}

static inline void
devd_transfer_service_put_file_cb (GObject      *object,
                                   GAsyncResult *result,
                                   gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  if (!devd_transfer_service_put_file_finish (DEVD_TRANSFER_SERVICE (object), result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);
}

static inline DexFuture *
devd_transfer_service_put_file (DevdTransferService   *service,
                                GFile                 *file,
                                const char            *remote_path,
                                GFileProgressCallback  progress,
                                gpointer               progress_data,
                                GDestroyNotify         progress_data_destroy)
{
  DexPromise *promise = dex_promise_new_cancellable ();
  devd_transfer_service_put_file_async (service,
                                        file,
                                        remote_path,
                                        progress,
                                        progress_data,
                                        progress_data_destroy,
                                        dex_promise_get_cancellable (promise),
                                        devd_transfer_service_put_file_cb,
                                        dex_ref (promise));
  return DEX_FUTURE (promise);
}

static inline void
devd_flatpak_service_install_bundle_cb (GObject      *object,
                                        GAsyncResult *result,
                                        gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  if (!devd_flatpak_service_install_bundle_finish (DEVD_FLATPAK_SERVICE (object), result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);
}

static inline DexFuture *
devd_flatpak_service_install_bundle (DevdFlatpakService    *service,
                                     const char            *remote_path)
{
  DexPromise *promise = dex_promise_new_cancellable ();
  devd_flatpak_service_install_bundle_async (service,
                                             remote_path,
                                             dex_promise_get_cancellable (promise),
                                             devd_flatpak_service_install_bundle_cb,
                                             dex_ref (promise));
  return DEX_FUTURE (promise);
}

G_END_DECLS
