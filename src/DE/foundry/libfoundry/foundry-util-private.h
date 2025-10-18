/* foundry-util-private.h
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

#include "foundry-debug.h"
#include "foundry-util.h"

G_BEGIN_DECLS

gboolean            _foundry_in_container                        (void);
const char * const *_foundry_host_environ                        (void);
char               *_foundry_create_host_triplet                 (const char     *arch,
                                                                  const char     *kernel,
                                                                  const char     *system);
char               *_foundry_get_shared_dir                      (void);
const char         *_foundry_get_system_type                     (void);
char               *_foundry_get_system_arch                     (void);
void                _foundry_fd_write_all                        (int             fd,
                                                                  const char     *message,
                                                                  gssize          to_write);
DexFuture          *_foundry_mkdtemp                             (const char     *tmpdir,
                                                                  const char     *template_name);
DexFuture          *_foundry_write_all_bytes                     (GOutputStream  *stream,
                                                                  GBytes        **bytesv,
                                                                  guint           n_bytesv);
DexFuture          *_foundry_flatten_list_model_new_from_futures (GPtrArray      *array);

static inline void
foundry_promise_resolve_bytes (DexPromise *promise,
                               GBytes     *bytes)
{
  const GValue gvalue = {G_TYPE_BYTES, {{.v_pointer = bytes}, {.v_int = 0}}};
  dex_promise_resolve (promise, &gvalue);
}

static inline DexFuture *
foundry_log_rejections (DexFuture *future,
                        gpointer   user_data)
{
  guint n_futures;

  dex_return_error_if_fail (DEX_IS_FUTURE_SET (future));

  n_futures = dex_future_set_get_size (DEX_FUTURE_SET (future));

  for (guint i = 0; i < n_futures; i++)
    {
      g_autoptr(GError) error = NULL;
      const GValue *value;

      if (!(value = dex_future_set_get_value_at (DEX_FUTURE_SET (future), i, &error)))
        g_warning ("Future %u of set failed: %s", i, error->message);
    }

  return dex_future_new_true ();
}

static inline gboolean
foundry_notify_pspec_in_main_cb (gpointer user_data)
{
  gpointer *data = user_data;

  g_object_notify_by_pspec (data[0], data[1]);
  g_clear_object (&data[0]);
  g_clear_pointer (&data[1], g_param_spec_unref);
  g_free (data);

  return G_SOURCE_REMOVE;
}

static inline void
foundry_notify_pspec_in_main (GObject    *object,
                              GParamSpec *pspec)
{
  gpointer *data;

  if G_LIKELY (FOUNDRY_IS_MAIN_THREAD ())
    {
      g_object_notify_by_pspec (object, pspec);
      return;
    }

  data = g_new (gpointer, 2);
  data[0] = g_object_ref (object);
  data[1] = g_param_spec_ref (pspec);

  g_idle_add_full (G_PRIORITY_LOW,
                   foundry_notify_pspec_in_main_cb,
                   data,
                   NULL);
}

static inline GWeakRef *
foundry_weak_ref_new (gpointer instance)
{
  GWeakRef *wr = g_new0 (GWeakRef, 1);
  g_weak_ref_init (wr, instance);
  return wr;
}

static inline void
foundry_weak_ref_free (GWeakRef *wr)
{
  g_weak_ref_clear (wr);
  g_free (wr);
}

typedef struct _FoundryWeakPair
{
  GWeakRef first;
  GWeakRef second;
} FoundryWeakPair;

static inline FoundryWeakPair *
foundry_weak_pair_new (gpointer first,
                       gpointer second)
{
  FoundryWeakPair *pair = g_new0 (FoundryWeakPair, 1);
  g_weak_ref_init (&pair->first, first);
  g_weak_ref_init (&pair->second, second);
  return pair;
}

static inline void
foundry_weak_pair_free (FoundryWeakPair *pair)
{
  g_weak_ref_clear (&pair->first);
  g_weak_ref_clear (&pair->second);
  g_free (pair);
}

static inline gboolean
foundry_weak_pair_get (FoundryWeakPair *pair,
                       gpointer         first,
                       gpointer         second)
{
  *(gpointer *)first = g_weak_ref_get (&pair->first);
  *(gpointer *)second = g_weak_ref_get (&pair->second);

  return *(gpointer *)first != NULL && *(gpointer *)second != NULL;
}

static inline void
_g_data_input_stream_read_line_utf8_cb (GObject      *object,
                                        GAsyncResult *result,
                                        gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  GError *error =  NULL;
  char *ret;
  gsize len;

  if ((ret = g_data_input_stream_read_line_finish_utf8 (G_DATA_INPUT_STREAM (object), result, &len, &error)))
    dex_promise_resolve_string (promise, g_steal_pointer (&ret));
  else
    dex_promise_reject (promise, g_steal_pointer (&error));
}

static inline DexFuture *
_g_data_input_stream_read_line_utf8 (GDataInputStream *stream)
{
  DexPromise *promise = dex_promise_new_cancellable ();
  g_data_input_stream_read_line_async (stream,
                                       G_PRIORITY_DEFAULT,
                                       dex_promise_get_cancellable (promise),
                                       _g_data_input_stream_read_line_utf8_cb,
                                       dex_ref (promise));
  return DEX_FUTURE (promise);
}

static inline void
_g_input_stream_read_bytes_cb (GObject      *object,
                               GAsyncResult *result,
                               gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  GError *error =  NULL;
  GBytes *ret;

  if ((ret = g_input_stream_read_bytes_finish (G_INPUT_STREAM (object), result, &error)))
    dex_promise_resolve_boxed (promise, G_TYPE_BYTES, g_steal_pointer (&ret));
  else
    dex_promise_reject (promise, g_steal_pointer (&error));
}

static inline DexFuture *
_g_input_stream_read_bytes (GInputStream *stream,
                            gsize         count)
{
  DexPromise *promise = dex_promise_new_cancellable ();
  g_input_stream_read_bytes_async (stream,
                                   count,
                                   G_PRIORITY_DEFAULT,
                                   dex_promise_get_cancellable (promise),
                                   _g_input_stream_read_bytes_cb,
                                   dex_ref (promise));
  return DEX_FUTURE (promise);
}

typedef struct _WeakRefGuard WeakRefGuard;

struct _WeakRefGuard
{
  gatomicrefcount ref_count;
  gpointer data;
};

static inline WeakRefGuard *
weak_ref_guard_new (gpointer data)
{
  WeakRefGuard *guard;

  guard = g_new0 (WeakRefGuard, 1);
  g_atomic_ref_count_init (&guard->ref_count);
  guard->data = data;

  return guard;
}

static inline WeakRefGuard *
weak_ref_guard_ref (WeakRefGuard *guard)
{
  g_atomic_ref_count_inc (&guard->ref_count);
  return guard;
}

static inline void
weak_ref_guard_unref (WeakRefGuard *guard)
{
  /* Always clear data pointer after first unref so that it
   * cannot be accessed unless both the expression/watch is
   * valid _and_ the weak ref is still active.
   */
  guard->data = NULL;

  if (g_atomic_ref_count_dec (&guard->ref_count))
    g_free (guard);
}

G_END_DECLS
