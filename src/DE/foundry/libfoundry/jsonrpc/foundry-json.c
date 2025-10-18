/* foundry-json.c
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

#include "foundry-json.h"

static DexFuture *
foundry_json_read_complete (DexFuture *completed,
                            gpointer   user_data)
{
  JsonParser *parser = user_data;
  g_autoptr(GInputStream) stream = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (JSON_IS_PARSER (parser));

  stream = dex_await_object (dex_ref (completed), NULL);
  g_assert (G_IS_INPUT_STREAM (stream));

  return foundry_json_parser_load_from_stream (parser, stream);
}

/**
 * foundry_json_parser_load_from_file:
 * @parser: a [class@Json.Parser]
 * @file: a [iface@Gio.File]
 *
 * Loads @file into @parser and returns a future that resolves
 * when that has completed.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to a boolean
 */
DexFuture *
foundry_json_parser_load_from_file (JsonParser *parser,
                                    GFile      *file)
{
  DexFuture *future;

  dex_return_error_if_fail (JSON_IS_PARSER (parser));
  dex_return_error_if_fail (G_IS_FILE (file));

  future = dex_file_read (file, G_PRIORITY_DEFAULT);
  future = dex_future_then (future,
                            foundry_json_read_complete,
                            g_object_ref (parser),
                            g_object_unref);

  return future;
}

static void
foundry_json_parser_load_from_stream_cb (GObject      *object,
                                         GAsyncResult *result,
                                         gpointer      user_data)
{
  JsonParser *parser = (JsonParser *)object;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (JSON_IS_PARSER (parser));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!json_parser_load_from_stream_finish (parser, result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);
}

/**
 * foundry_json_parser_load_from_stream:
 * @parser: a [class@Json.Parser]
 * @stream: a [class@Gio.InputStream]
 *
 * Like json_parser_load_from_stream() but asynchronous and returns
 * a [class@Dex.Future] which can be awaited upon.
 *
 * Returns: (transfer full): a [class@Dex.Future].
 */
DexFuture *
foundry_json_parser_load_from_stream (JsonParser   *parser,
                                      GInputStream *stream)
{
  DexPromise *promise;

  dex_return_error_if_fail (JSON_IS_PARSER (parser));
  dex_return_error_if_fail (G_IS_INPUT_STREAM (stream));

  promise = dex_promise_new_cancellable ();

  json_parser_load_from_stream_async (parser,
                                      stream,
                                      dex_promise_get_cancellable (promise),
                                      foundry_json_parser_load_from_stream_cb,
                                      dex_ref (promise));

  return DEX_FUTURE (promise);
}

const char *
foundry_json_node_get_string_at (JsonNode     *node,
                                 const char   *first_key,
                                 ...)
{
  const char *key = first_key;
  va_list args;

  if (node == NULL)
    return NULL;

  va_start (args, first_key);

  while (node != NULL && key != NULL)
    {
      JsonObject *object;

      if (!JSON_NODE_HOLDS_OBJECT (node))
        {
          node = NULL;
          break;
        }

      object = json_node_get_object (node);

      if (!json_object_has_member (object, key))
        {
          node = NULL;
          break;
        }

      node = json_object_get_member (object, key);
      key = va_arg (args, const char *);
    }

  va_end (args);

  if (node != NULL && JSON_NODE_HOLDS_VALUE (node))
    return json_node_get_string (node);

  return NULL;
}

typedef struct _JsonNodeFromBytes
{
  DexPromise *promise;
  GBytes *bytes;
} JsonNodeFromBytes;

static void
foundry_json_node_from_bytes_worker (gpointer data)
{
  JsonNodeFromBytes *state = data;
  g_autoptr(JsonParser) parser = json_parser_new ();
  g_autoptr(GError) error = NULL;

  if (!json_parser_load_from_data (parser,
                                   g_bytes_get_data (state->bytes, NULL),
                                   g_bytes_get_size (state->bytes),
                                   &error))
    dex_promise_reject (state->promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boxed (state->promise,
                               JSON_TYPE_NODE,
                               json_parser_steal_root (parser));

  dex_clear (&state->promise);
  g_clear_pointer (&state->bytes, g_bytes_unref);
  g_free (state);
}

/**
 * foundry_json_node_from_bytes:
 * @bytes: a [struct@GLib.Bytes]
 *
 * Bytes to be deocded into a json node
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [struct@Json.Node] or rejects with error.
 */
DexFuture *
foundry_json_node_from_bytes (GBytes *bytes)
{
  DexPromise *promise = dex_promise_new_cancellable ();
  JsonNodeFromBytes *state = g_new0 (JsonNodeFromBytes, 1);

  state->promise = dex_ref (promise);
  state->bytes = g_bytes_ref (bytes);

  dex_scheduler_push (dex_thread_pool_scheduler_get_default (),
                      foundry_json_node_from_bytes_worker,
                      state);

  return DEX_FUTURE (promise);
}

/**
 * foundry_json_node_new_strv:
 * @strv:
 *
 * Returns: (transfer full):
 */
JsonNode *
foundry_json_node_new_strv (const char * const *strv)
{
  g_autoptr(JsonArray) ar = NULL;
  JsonNode *node;

  if (strv == NULL)
    return json_node_new (JSON_NODE_NULL);

  node = json_node_new (JSON_NODE_ARRAY);
  ar = json_array_new ();

  for (gsize i = 0; strv[i]; i++)
    json_array_add_string_element (ar, strv[i]);

  json_node_set_array (node, ar);

  return node;
}

static void
foundry_json_node_to_bytes_worker (gpointer data)
{
  gpointer *state = data;
  JsonNode *node = state[0];
  DexPromise *promise = state[1];
  g_autoptr(JsonGenerator) generator = json_generator_new ();
  g_autofree char *contents = NULL;
  gsize len;

  json_generator_set_root (generator, node);
  contents = json_generator_to_data (generator, &len);

  dex_promise_resolve_boxed (promise,
                             G_TYPE_BYTES,
                             g_bytes_new (g_steal_pointer (&contents), len));

  g_clear_pointer (&state[0], json_node_unref);
  dex_clear (&state[1]);
  g_free (state);
}

/**
 * foundry_json_node_to_bytes:
 * @node:
 *
 * @node must not be modified after calling this function
 * until the future as resolved or rejected.
 *
 * Returns: (transfer full): a future that resolves to a
 *   [struct@GLib.Bytes] or rejects with error.
 */
DexFuture *
foundry_json_node_to_bytes (JsonNode *node)
{
  DexPromise *promise;
  gpointer *state;

  dex_return_error_if_fail (node != NULL);

  promise = dex_promise_new_cancellable ();

  state = g_new0 (gpointer, 2);
  state[0] = json_node_ref (node);
  state[1] = dex_ref (promise);

  dex_scheduler_push (dex_thread_pool_scheduler_get_default (),
                      foundry_json_node_to_bytes_worker,
                      state);

  return DEX_FUTURE (promise);
}
