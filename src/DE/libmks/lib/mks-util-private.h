/*
 * mks-util-private.h
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

#include <cairo.h>
#include <gdk/gdk.h>
#include <gio/gio.h>

G_BEGIN_DECLS

#define _CAIRO_CHECK_VERSION(major, minor, micro) \
  (CAIRO_VERSION_MAJOR > (major) || \
   (CAIRO_VERSION_MAJOR == (major) && CAIRO_VERSION_MINOR > (minor)) || \
   (CAIRO_VERSION_MAJOR == (major) && CAIRO_VERSION_MINOR == (minor) && \
    CAIRO_VERSION_MICRO >= (micro)))

#ifdef MKS_DEBUG
# define MKS_ENTRY      G_STMT_START { g_debug("ENTRY: %s():%u", G_STRFUNC, __LINE__); } G_STMT_END
# define MKS_EXIT       G_STMT_START { g_debug(" EXIT: %s():%u", G_STRFUNC, __LINE__); return; } G_STMT_END
# define MKS_RETURN(_r) G_STMT_START { typeof(_r) __ret = (_r); g_debug(" EXIT: %s():%u", G_STRFUNC, __LINE__); return __ret; } G_STMT_END
#else
# define MKS_ENTRY      G_STMT_START { } G_STMT_END
# define MKS_EXIT       G_STMT_START { return; } G_STMT_END
# define MKS_RETURN(_r) G_STMT_START { typeof(_r) __ret = (_r); return (__ret); } G_STMT_END
#endif


gboolean         mks_socketpair_create                (int     *us,
                                                       int     *them,
                                                       GError **error);
gboolean         mks_scroll_event_is_inverted         (GdkEvent              *event);
void             mks_socketpair_connection_new        (GDBusConnectionFlags   flags,
                                                       GCancellable          *cancellable,
                                                       GAsyncReadyCallback    callback,
                                                       gpointer               user_data);
GDBusConnection *mks_socketpair_connection_new_finish (GAsyncResult          *result,
                                                       int                   *peer_fd,
                                                       GError               **error);

G_END_DECLS
