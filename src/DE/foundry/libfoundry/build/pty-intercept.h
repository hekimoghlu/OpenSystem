/* ide-pty-intercept.h
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

#include <unistd.h>

G_BEGIN_DECLS

#define PTY_FD_INVALID (-1)
#define PTY_INTERCEPT_MAGIC (0x81723647)
#define IS_PTY_INTERCEPT(s) ((s) != NULL && (s)->magic == PTY_INTERCEPT_MAGIC)

typedef struct _PtyIntercept PtyIntercept;
typedef struct _PtyInterceptSide PtyInterceptSide;

typedef void (*PtyInterceptCallback) (const PtyIntercept     *intercept,
                                      const PtyInterceptSide *side,
                                      const guint8           *data,
                                      gsize                   len,
                                      gpointer                user_data);

struct _PtyInterceptSide
{
  GIOChannel           *channel;
  guint                 in_watch;
  guint                 out_watch;
  int                   read_prio;
  int                   write_prio;
  GBytes               *out_bytes;
  PtyInterceptCallback  callback;
  gpointer              callback_data;
};

struct _PtyIntercept
{
  gsize            magic;
  PtyInterceptSide consumer;
  PtyInterceptSide producer;
};

int      pty_intercept_create_consumer (void);
int      pty_intercept_create_producer (int                   consumer_fd,
                                        gboolean              blocking);
gboolean pty_intercept_init            (PtyIntercept         *self,
                                        int                   fd,
                                        GMainContext         *main_context);
int      pty_intercept_get_fd          (PtyIntercept         *self);
gboolean pty_intercept_set_size        (PtyIntercept         *self,
                                        guint                 rows,
                                        guint                 columns);
void     pty_intercept_clear           (PtyIntercept         *self);
void     pty_intercept_set_callback    (PtyIntercept         *self,
                                        PtyInterceptSide     *side,
                                        PtyInterceptCallback  callback,
                                        gpointer              user_data);

G_END_DECLS
