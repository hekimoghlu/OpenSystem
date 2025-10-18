/*
 * mks-session.h
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#if !defined(MKS_INSIDE) && !defined(MKS_COMPILATION)
# error "Only <libmks.h> can be included directly."
#endif

#include <gio/gio.h>

#include "mks-types.h"
#include "mks-version-macros.h"

G_BEGIN_DECLS

#define MKS_TYPE_SESSION (mks_session_get_type())

MKS_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (MksSession, mks_session, MKS, SESSION, GObject)

MKS_AVAILABLE_IN_ALL
void mks_session_new_for_connection                         (GDBusConnection     *connection,
                                                             int                  io_priority,
                                                             GCancellable        *cancellable,
                                                             GAsyncReadyCallback  callback,
                                                             gpointer             user_data);

MKS_AVAILABLE_IN_ALL
MksSession *mks_session_new_for_connection_finish           (GAsyncResult         *result,
                                                             GError              **error);

MKS_AVAILABLE_IN_ALL
void mks_session_new_for_connection_with_name               (GDBusConnection     *connection,
                                                             const char          *bus_name,
                                                             int                  io_priority,
                                                             GCancellable        *cancellable,
                                                             GAsyncReadyCallback  callback,
                                                             gpointer             user_data);

MKS_AVAILABLE_IN_ALL
MksSession *mks_session_new_for_connection_with_name_finish (GAsyncResult         *result,
                                                             GError              **error);

MKS_AVAILABLE_IN_ALL
MksSession *mks_session_new_for_connection_sync             (GDBusConnection  *connection,
                                                             GCancellable     *cancellable,
                                                             GError          **error);

MKS_AVAILABLE_IN_ALL
MksSession *mks_session_new_for_connection_with_name_sync   (GDBusConnection  *connection,
                                                             const char       *bus_name,
                                                             GCancellable     *cancellable,
                                                             GError          **error);
MKS_AVAILABLE_IN_ALL
GDBusConnection *mks_session_get_connection                 (MksSession           *self);
MKS_AVAILABLE_IN_ALL
GListModel      *mks_session_get_devices                    (MksSession           *self);
MKS_AVAILABLE_IN_ALL
const char      *mks_session_get_name                       (MksSession           *self);
MKS_AVAILABLE_IN_ALL
const char      *mks_session_get_uuid                       (MksSession           *self);
MKS_AVAILABLE_IN_ALL
const char      *mks_session_get_bus_name                   (MksSession           *self);
MKS_AVAILABLE_IN_ALL
MksScreen       *mks_session_ref_screen                     (MksSession           *self);

G_END_DECLS
