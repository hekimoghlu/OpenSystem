/* foundry-json-input-stream.c
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
#include "foundry-json-input-stream-private.h"

struct _FoundryJsonInputStream
{
  GDataInputStream parent_instance;
  int max_size_bytes;
};

G_DEFINE_FINAL_TYPE (FoundryJsonInputStream, foundry_json_input_stream, G_TYPE_DATA_INPUT_STREAM)

G_DEFINE_QUARK (foundry-json-error, foundry_json_error)

static gboolean debug_enabled;

static void
foundry_json_input_stream_class_init (FoundryJsonInputStreamClass *klass)
{
  debug_enabled = g_getenv ("JSONRPC_DEBUG") != NULL;
}

static void
foundry_json_input_stream_init (FoundryJsonInputStream *self)
{
  self->max_size_bytes = 16 * 1024 * 1024;
}

static DexFuture *
foundry_json_input_stream_deserialize_cb (DexFuture *completed,
                                          gpointer   user_data)
{
  DexPromise *promise = user_data;
  g_autoptr(JsonNode) node = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (DEX_IS_PROMISE (promise));

  if (!(node = dex_await_boxed (dex_ref (completed), &error)))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boxed (promise, JSON_TYPE_NODE, g_steal_pointer (&node));

  return dex_future_new_true ();
}

static void
foundry_json_input_stream_read_upto_cb (GObject      *object,
                                        GAsyncResult *result,
                                        gpointer      user_data)
{
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;
  g_autofree char *contents = NULL;
  gsize len;

  g_assert (FOUNDRY_IS_JSON_INPUT_STREAM (object));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!(contents = g_data_input_stream_read_upto_finish (G_DATA_INPUT_STREAM (object), result, &len, &error)))
    {
      if (error != NULL)
        dex_promise_reject (promise, g_steal_pointer (&error));
      else
        dex_promise_reject (promise,
                            g_error_new (FOUNDRY_JSON_ERROR,
                                         FOUNDRY_JSON_ERROR_EOF,
                                         "End of Stream"));
    }
  else
    {
      g_autoptr(GBytes) bytes = NULL;

      if (debug_enabled)
        FOUNDRY_DUMP_BYTES (read, contents, len);

      bytes = g_bytes_new_take (g_steal_pointer (&contents), len);
      dex_future_disown (dex_future_finally (foundry_json_node_from_bytes (bytes),
                                             foundry_json_input_stream_deserialize_cb,
                                             dex_ref (promise),
                                             dex_unref));
    }
}

/**
 * foundry_json_input_stream_read_upto:
 * @self: a [class@Foundry.JsonInputStream]
 * @stop_chars: the characters that delimit json messages, such as \n.
 * @stop_chars_len: then length of @stop_chars or -1 if it is \0 terminated
 *
 * Reads the next JSON message from the stream.
 *
 * If stop_chars_len is > 0, then you man use `\0` as a delimiter.
 *
 * Use this form when you do not have HTTP headers containing the content
 * length of the JSON message.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [struct@Json.Node] or rejects with error.
 */
DexFuture *
foundry_json_input_stream_read_upto (FoundryJsonInputStream *self,
                                     const char             *stop_chars,
                                     gssize                  stop_chars_len)
{
  DexPromise *promise;

  dex_return_error_if_fail (FOUNDRY_IS_JSON_INPUT_STREAM (self));

  promise = dex_promise_new_cancellable ();
  g_data_input_stream_read_upto_async (G_DATA_INPUT_STREAM (self),
                                       stop_chars,
                                       stop_chars_len,
                                       G_PRIORITY_DEFAULT,
                                       dex_promise_get_cancellable (promise),
                                       foundry_json_input_stream_read_upto_cb,
                                       dex_ref (promise));
  return DEX_FUTURE (promise);
}

static void
read_upto_cb (GObject      *object,
              GAsyncResult *result,
              gpointer      user_data)
{
  DexPromise *promise = user_data;
  g_autoptr(GError) error = NULL;
  g_autofree char *contents = NULL;
  gsize len = 0;

  g_assert (FOUNDRY_IS_JSON_INPUT_STREAM (object));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  contents = g_data_input_stream_read_upto_finish (G_DATA_INPUT_STREAM (object), result, &len, &error);

  if (contents == NULL && error != NULL)
    {
      dex_promise_reject (promise, g_steal_pointer (&error));
    }
  else if (contents == NULL)
    {
      dex_promise_resolve_string (promise, g_strdup (""));
    }
  else
    {
      if G_UNLIKELY (debug_enabled)
        FOUNDRY_DUMP_BYTES (read, contents, len);

      if (!g_utf8_validate_len (contents, len, NULL))
        dex_promise_reject (promise,
                            g_error_new (G_IO_ERROR,
                                         G_IO_ERROR_INVALID_DATA,
                                         "Invalid UTF-8"));
      else
        dex_promise_resolve_string (promise, g_steal_pointer (&contents));
    }
}

static DexFuture *
read_upto (FoundryJsonInputStream *self,
           const char             *stop_chars,
           gsize                   stop_chars_len)
{
  DexPromise *promise;

  g_assert (FOUNDRY_IS_JSON_INPUT_STREAM (self));

  promise = dex_promise_new_cancellable ();
  g_data_input_stream_read_upto_async (G_DATA_INPUT_STREAM (self),
                                       stop_chars,
                                       stop_chars_len,
                                       G_PRIORITY_DEFAULT,
                                       dex_promise_get_cancellable (promise),
                                       read_upto_cb,
                                       dex_ref (promise));
  return DEX_FUTURE (promise);
}

static DexFuture *
foundry_json_input_stream_read_fiber (gpointer user_data)
{
  FoundryJsonInputStream *self = user_data;
  g_autoptr(GBytes) bytes = NULL;
  g_autoptr(GError) error = NULL;
  gint64 content_length = -1;

  g_assert (FOUNDRY_IS_JSON_INPUT_STREAM (self));

  for (;;)
    {
      g_autofree char *line = dex_await_string (read_upto (self, "\n", 1), &error);
      g_autofree char *key = NULL;
      g_autofree char *value = NULL;
      const char *colon;

      if (line == NULL)
        return dex_future_new_for_error (g_steal_pointer (&error));

      if (line[0] == 0 || line[0] == '\r')
        goto skip;

      if (!(colon = strchr (line, ':')))
        return dex_future_new_reject (G_IO_ERROR,
                                      G_IO_ERROR_INVALID_DATA,
                                      "Expected HTTP header but got other data");

      key = g_strndup (line, colon - line);
      value = g_strstrip (g_strdup (colon + 1));

      if (strncasecmp (key, "Content-Length", 16) == 0)
        {
          content_length = g_ascii_strtoll (value, NULL, 10);

          if (((content_length == G_MININT64 || content_length == G_MAXINT64) && errno == ERANGE) ||
              (content_length < 0) ||
              (content_length == G_MAXSSIZE) ||
              (content_length > (gint64)self->max_size_bytes))
            return dex_future_new_reject (G_IO_ERROR,
                                          G_IO_ERROR_INVALID_DATA,
                                          "Invalid Content-Length provided");
        }

    skip:
      if (!dex_await (dex_input_stream_skip (G_INPUT_STREAM (self), 1, G_PRIORITY_DEFAULT), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));

      if (key == NULL)
        break;
    }

  if (content_length < 0)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_INVALID_DATA,
                                  "Content-Length was not provided");

  if (!(bytes = dex_await_boxed (dex_input_stream_read_bytes (G_INPUT_STREAM (self),
                                                              (gsize)content_length,
                                                              G_PRIORITY_DEFAULT),
                                 &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if G_UNLIKELY (debug_enabled)
    FOUNDRY_DUMP_BYTES (read,
                        ((const char *)g_bytes_get_data (bytes, NULL)),
                        (g_bytes_get_size (bytes)));

  return foundry_json_node_from_bytes (bytes);
}

/**
 * foundry_json_input_stream_read_http:
 * @self: a [class@Foundry.JsonInputStream]
 *
 * Reads the next message expecting a HTTP header at the start of
 * the stream.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [struct@Json.Node] or rejects with error
 */
DexFuture *
foundry_json_input_stream_read_http (FoundryJsonInputStream *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_JSON_INPUT_STREAM (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_json_input_stream_read_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

FoundryJsonInputStream *
foundry_json_input_stream_new (GInputStream *base_stream,
                               gboolean      close_base_stream)
{
  g_return_val_if_fail (G_IS_INPUT_STREAM (base_stream), NULL);

  return g_object_new (FOUNDRY_TYPE_JSON_INPUT_STREAM,
                       "base-stream", base_stream,
                       "close-base-stream", close_base_stream,
                       NULL);
}
