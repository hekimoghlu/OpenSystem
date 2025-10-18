/* foundry-json-output-stream.c
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

#include "foundry-debug.h"
#include "foundry-json.h"
#include "foundry-json-output-stream-private.h"
#include "foundry-util-private.h"

struct _FoundryJsonOutputStream
{
  GDataOutputStream parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundryJsonOutputStream, foundry_json_output_stream, G_TYPE_DATA_OUTPUT_STREAM)

static GBytes *null_bytes;
static gboolean debug_enabled;

static void
foundry_json_output_stream_class_init (FoundryJsonOutputStreamClass *klass)
{
  null_bytes = g_bytes_new_static ("", 0);
  debug_enabled = g_getenv ("JSONRPC_DEBUG") != NULL;
}

static void
foundry_json_output_stream_init (FoundryJsonOutputStream *self)
{
}

typedef struct _Write
{
  GOutputStream *stream;
  GBytes        *delimiter;
  GString       *headers;
} Write;

static void
write_free (Write *state)
{
  g_clear_object (&state->stream);
  g_clear_pointer (&state->delimiter, g_bytes_unref);
  if (state->headers)
    g_string_free (state->headers, TRUE), state->headers = NULL;
  g_free (state);
}

static DexFuture *
foundry_json_output_stream_serialize_cb (DexFuture *completed,
                                         gpointer   user_data)
{
  Write *state = user_data;
  g_autoptr(GBytes) bytes = NULL;
  g_autoptr(GBytes) headers = NULL;
  GBytes *to_write[3];
  guint nbytes = 0;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_JSON_OUTPUT_STREAM (state->stream));
  g_assert (state->delimiter != NULL);

  bytes = dex_await_boxed (dex_ref (completed), NULL);

  g_assert (bytes != NULL);

  if (state->headers != NULL)
    {
      gsize msglen = g_bytes_get_size (bytes) + g_bytes_get_size (state->delimiter);

      g_string_append_printf (state->headers,
                              "Content-Length: %"G_GSIZE_FORMAT"\r\n\r\n", msglen);
      headers = g_string_free_to_bytes (g_steal_pointer (&state->headers));
    }

  if (headers && g_bytes_get_size (headers) > 0)
    to_write[nbytes++] = headers;

  to_write[nbytes++] = bytes;

  if (state->delimiter && g_bytes_get_size (state->delimiter) > 0)
    to_write[nbytes++] = state->delimiter;

  if G_UNLIKELY (debug_enabled)
    {
      for (guint j = 0; j < nbytes; j++)
        {
          FOUNDRY_DUMP_BYTES (write,
                              ((const char *)g_bytes_get_data (to_write[j], NULL)),
                              (g_bytes_get_size (to_write[j])));
        }
    }

  return _foundry_write_all_bytes (state->stream, to_write, nbytes);
}

/**
 * foundry_json_output_stream_write:
 * @self: a [class@Foundry.JsonOutputStream]
 * @headers: (nullable): a hashtable of headers to write to the stream
 * @node: the JSON node to be written
 * @delimiter: the delimiter to use as the message suffix
 *
 * The caller must not mutate @node after calling this function
 * until after the operation has completed.
 *
 * If @headers is non-`null`, then they will be added to the stream
 * at the start in HTTP style. The `Content-Length` will automatically
 * be added followed by `\r\n\r\n`.
 *
 * If you only want `Content-Length` provided, then it is okay to
 * pass an empty `GHashTable`.
 *
 * If @headers is `null`, then no headers will be written.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 */
DexFuture *
foundry_json_output_stream_write (FoundryJsonOutputStream *self,
                                  GHashTable              *headers,
                                  JsonNode                *node,
                                  GBytes                  *delimiter)
{
  Write *state;

  dex_return_error_if_fail (FOUNDRY_IS_JSON_OUTPUT_STREAM (self));
  dex_return_error_if_fail (node != NULL);

  if (delimiter == NULL)
    delimiter = null_bytes;

  state = g_new0 (Write, 1);
  state->delimiter = g_bytes_ref (delimiter);
  state->stream = g_object_ref (G_OUTPUT_STREAM (self));

  if (headers != NULL)
    {
      GString *str = g_string_new (NULL);
      GHashTableIter iter;
      gpointer key, value;

      g_hash_table_iter_init (&iter, headers);
      while (g_hash_table_iter_next (&iter, &key, &value))
        {
          g_string_append (str, key);
          g_string_append_c (str, ':');
          g_string_append_c (str, ' ');
          g_string_append (str, value);
          g_string_append (str, "\r\n");
        }

      state->headers = g_steal_pointer (&str);
    }

  return dex_future_then (foundry_json_node_to_bytes (node),
                          foundry_json_output_stream_serialize_cb,
                          state,
                          (GDestroyNotify) write_free);
}

FoundryJsonOutputStream *
foundry_json_output_stream_new (GOutputStream *base_stream,
                                gboolean       close_base_stream)
{
  g_return_val_if_fail (G_IS_OUTPUT_STREAM (base_stream), NULL);

  return g_object_new (FOUNDRY_TYPE_JSON_OUTPUT_STREAM,
                       "base-stream", base_stream,
                       "close-base-stream", close_base_stream,
                       NULL);
}
