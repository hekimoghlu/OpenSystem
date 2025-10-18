/* foundry-jsonrpc-driver.c
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

#include <json-glib/json-glib.h>

#include "foundry-json-input-stream-private.h"
#include "foundry-json-output-stream-private.h"
#include "foundry-jsonrpc-driver-private.h"
#include "foundry-jsonrpc-waiter-private.h"
#include "foundry-util-private.h"

struct _FoundryJsonrpcDriver
{
  GObject                  parent_instance;
  GIOStream               *stream;
  FoundryJsonInputStream  *input;
  FoundryJsonOutputStream *output;
  DexChannel              *output_channel;
  GHashTable              *requests;
  GBytes                  *delimiter;
  gint64                   last_seq;
  FoundryJsonrpcStyle      style : 2;
};

enum {
  PROP_0,
  PROP_STREAM,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryJsonrpcDriver, foundry_jsonrpc_driver, G_TYPE_OBJECT)

enum {
  SIGNAL_HANDLE_METHOD_CALL,
  SIGNAL_HANDLE_NOTIFICATION,
  N_SIGNALS
};

static GParamSpec *properties[N_PROPS];
static guint signals[N_SIGNALS];
static GHashTable *empty_headers;

static JsonNode *
get_next_id (FoundryJsonrpcDriver *self)
{
  gint64 seq = ++self->last_seq;
  JsonNode *node = json_node_new (JSON_NODE_VALUE);
  json_node_set_int (node, seq);
  return node;
}

static gboolean
check_string (JsonNode   *node,
              const char *value)
{
  if (node == NULL)
    return FALSE;

  if (!JSON_NODE_HOLDS_VALUE (node))
    return FALSE;

  return g_strcmp0 (value, json_node_get_string (node)) == 0;
}

static gboolean
is_jsonrpc (JsonNode *node)
{
  JsonObject *obj;

  return JSON_NODE_HOLDS_OBJECT (node) &&
         (obj = json_node_get_object (node)) &&
         json_object_has_member (obj, "jsonrpc") &&
         check_string (json_object_get_member (obj, "jsonrpc"), "2.0");
}

static gboolean
is_jsonrpc_notification (JsonNode *node)
{
  JsonObject *obj;

  return JSON_NODE_HOLDS_OBJECT (node) &&
         (obj = json_node_get_object (node)) &&
         !json_object_has_member (obj, "id") &&
         json_object_has_member (obj, "method");
}

static gboolean
is_jsonrpc_method_call (JsonNode *node)
{
  JsonObject *obj;

  return JSON_NODE_HOLDS_OBJECT (node) &&
         (obj = json_node_get_object (node)) &&
         json_object_has_member (obj, "id") &&
         json_object_has_member (obj, "method") &&
         json_object_has_member (obj, "params");
}

static gboolean
is_jsonrpc_method_reply (JsonNode *node)
{
  JsonObject *obj;

  return JSON_NODE_HOLDS_OBJECT (node) &&
         (obj = json_node_get_object (node)) &&
         json_object_has_member (obj, "id") &&
         (json_object_has_member (obj, "result") ||
          json_object_has_member (obj, "error"));
}

static void
foundry_jsonrpc_driver_handle_message (FoundryJsonrpcDriver *self,
                                       JsonNode             *node)
{
  g_assert (FOUNDRY_IS_JSONRPC_DRIVER (self));
  g_assert (node != NULL);

  if (JSON_NODE_HOLDS_ARRAY (node))
    {
      JsonArray *ar = json_node_get_array (node);
      gsize len = json_array_get_length (ar);

      for (gsize i = 0; i < len; i++)
        {
          JsonNode *child = json_array_get_element (ar, i);

          foundry_jsonrpc_driver_handle_message (self, child);
        }

      return;
    }

  if (is_jsonrpc (node))
    {
      if (is_jsonrpc_notification (node))
        {
          JsonObject *obj = json_node_get_object (node);
          const char *method = json_object_get_string_member (obj, "method");
          JsonNode *params = json_object_get_member (obj, "params");

          g_signal_emit (self, signals[SIGNAL_HANDLE_NOTIFICATION], 0, method, params);

          return;
        }

      if (is_jsonrpc_method_reply (node))
        {
          JsonObject *obj = json_node_get_object (node);
          JsonNode *id = json_object_get_member (obj, "id");
          JsonNode *result = json_object_get_member (obj, "result");
          JsonNode *error = json_object_get_member (obj, "error");
          g_autoptr(JsonNode) stolen_key = NULL;
          g_autoptr(FoundryJsonrpcWaiter) waiter = NULL;

          if (g_hash_table_steal_extended (self->requests,
                                           id,
                                           (gpointer *)&stolen_key,
                                           (gpointer *)&waiter))
            {
              if (error != NULL && JSON_NODE_HOLDS_OBJECT (error))
                {
                  JsonObject *err = json_node_get_object (error);
                  const char *message = json_object_get_string_member (err, "message");
                  gint64 code = json_object_get_int_member (err, "code");

                  foundry_jsonrpc_waiter_reject (waiter,
                                                 g_error_new_literal (g_quark_from_static_string ("foundry-jsonrpc-error"),
                                                                      code,
                                                                      message ? message : "unknown error"));
                }
              else
                {
                  foundry_jsonrpc_waiter_reply (waiter, json_node_ref (result));
                }
            }

          return;
        }

      if (is_jsonrpc_method_call (node))
        {
          JsonObject *obj = json_node_get_object (node);
          const char *method = json_object_get_string_member (obj, "method");
          JsonNode *params = json_object_get_member (obj, "params");
          JsonNode *id = json_object_get_member (obj, "id");
          gboolean ret = FALSE;

          g_signal_emit (self, signals[SIGNAL_HANDLE_METHOD_CALL], 0, method, params, id, &ret);

          if (ret == FALSE)
            foundry_jsonrpc_driver_reply_with_error (self, id, -32601, "Method not found");

          return;
        }
    }

  /* Protocol error */
  g_io_stream_close_async (self->stream, 0, NULL, NULL, NULL);
}

static char *
get_id_as_string (JsonNode *node)
{
  if (node == NULL)
    return NULL;

  if (!JSON_NODE_HOLDS_VALUE (node))
    return g_strdup_printf ("%p", node);

  if (json_node_get_value_type (node) == G_TYPE_STRING)
    return g_strdup (json_node_get_string (node));

  return g_strdup_printf ("%"G_GINT64_FORMAT, json_node_get_int (node));
}

static guint
node_hash (gconstpointer a)
{
  g_autofree char *str = get_id_as_string ((JsonNode *)a);
  return g_str_hash (str);
}

static gboolean
node_equal (gconstpointer a,
            gconstpointer b)
{
  g_autofree char *str_a = get_id_as_string ((JsonNode *)a);
  g_autofree char *str_b = get_id_as_string ((JsonNode *)b);

  return g_strcmp0 (str_a, str_b) == 0;
}

static void
foundry_jsonrpc_driver_dispose (GObject *object)
{
  FoundryJsonrpcDriver *self = (FoundryJsonrpcDriver *)object;

  if (self->output_channel)
    {
      dex_channel_close_send (self->output_channel);
      dex_channel_close_receive (self->output_channel);
      dex_clear (&self->output_channel);
    }

  foundry_jsonrpc_driver_stop (self);

  g_clear_object (&self->input);
  g_clear_object (&self->output);
  g_clear_object (&self->stream);

  g_clear_pointer (&self->requests, g_hash_table_unref);
  g_clear_pointer (&self->delimiter, g_bytes_unref);

  G_OBJECT_CLASS (foundry_jsonrpc_driver_parent_class)->dispose (object);
}

static void
foundry_jsonrpc_driver_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryJsonrpcDriver *self = FOUNDRY_JSONRPC_DRIVER (object);

  switch (prop_id)
    {
    case PROP_STREAM:
      g_value_set_object (value, self->stream);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_jsonrpc_driver_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  FoundryJsonrpcDriver *self = FOUNDRY_JSONRPC_DRIVER (object);

  switch (prop_id)
    {
    case PROP_STREAM:
      self->stream = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_jsonrpc_driver_class_init (FoundryJsonrpcDriverClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_jsonrpc_driver_dispose;
  object_class->get_property = foundry_jsonrpc_driver_get_property;
  object_class->set_property = foundry_jsonrpc_driver_set_property;

  properties[PROP_STREAM] =
    g_param_spec_object ("stream", NULL, NULL,
                         G_TYPE_IO_STREAM,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  /**
   * FoundryJsonrpcDriver::handle-method-call:
   * @self: a [class@Foundry.JsonrpcDriver]
   * @method:
   * @params: (nullable):
   * @id:
   *
   * Returns: %TRUE if handled; otherwise %FALSE
   */
  signals[SIGNAL_HANDLE_METHOD_CALL] =
    g_signal_new ("handle-method-call",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  g_signal_accumulator_first_wins, NULL,
                  NULL,
                  G_TYPE_BOOLEAN, 3, G_TYPE_STRING, JSON_TYPE_NODE, JSON_TYPE_NODE);

  signals[SIGNAL_HANDLE_NOTIFICATION] =
    g_signal_new ("handle-notification",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 2, G_TYPE_STRING, JSON_TYPE_NODE);

  empty_headers = g_hash_table_new (NULL, NULL);
}

static void
foundry_jsonrpc_driver_init (FoundryJsonrpcDriver *self)
{
  self->output_channel = dex_channel_new (0);
  self->requests = g_hash_table_new_full (node_hash,
                                          node_equal,
                                          (GDestroyNotify)json_node_unref,
                                          g_object_unref);
}

FoundryJsonrpcDriver *
foundry_jsonrpc_driver_new (GIOStream           *stream,
                            FoundryJsonrpcStyle  style)
{
  FoundryJsonrpcDriver *self;
  GInputStream *input;
  GOutputStream *output;

  g_return_val_if_fail (G_IS_IO_STREAM (stream), NULL);
  g_return_val_if_fail (style > 0, NULL);
  g_return_val_if_fail (style <= FOUNDRY_JSONRPC_STYLE_NIL, NULL);

  self = g_object_new (FOUNDRY_TYPE_JSONRPC_DRIVER,
                       "stream", stream,
                       NULL);

  self->style = style;

  if (style == FOUNDRY_JSONRPC_STYLE_LF)
    self->delimiter = g_bytes_new ("\n", 1);
  else if (style == FOUNDRY_JSONRPC_STYLE_NIL)
    self->delimiter = g_bytes_new ("\0", 1);

  input = g_io_stream_get_input_stream (stream);
  output = g_io_stream_get_output_stream (stream);

  self->input = foundry_json_input_stream_new (input, TRUE);
  self->output = foundry_json_output_stream_new (output, TRUE);

  return self;
}

/**
 * foundry_jsonrpc_driver_call:
 * @self: a [class@Foundry.JsonrpcDriver]
 *
 * Returns: (transfer full): a future that resolves to a [struct@Json.Node]
 *   containing the reply.
 */
DexFuture *
foundry_jsonrpc_driver_call (FoundryJsonrpcDriver *self,
                             const char           *method,
                             JsonNode             *params)
{
  g_autoptr(FoundryJsonrpcWaiter) waiter = NULL;
  g_autoptr(JsonObject) object = NULL;
  g_autoptr(JsonNode) node = NULL;
  g_autoptr(JsonNode) id = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_JSONRPC_DRIVER (self));
  dex_return_error_if_fail (method != NULL);

  id = get_next_id (self);

  object = json_object_new ();

  json_object_set_string_member (object, "jsonrpc", "2.0");
  json_object_set_member (object, "id", json_node_ref (id));
  json_object_set_string_member (object, "method", method);

  if (params != NULL)
    json_object_set_member (object, "params", json_node_ref (params));
  else
    json_object_set_null_member (object, "params");

  node = json_node_new (JSON_NODE_OBJECT);
  json_node_set_object (node, object);

  waiter = foundry_jsonrpc_waiter_new (node, id);

  g_hash_table_replace (self->requests,
                        json_node_ref (id),
                        g_object_ref (waiter));

  dex_future_disown (dex_future_catch (dex_channel_send (self->output_channel,
                                                         dex_future_new_take_object (g_object_ref (waiter))),
                                       foundry_jsonrpc_waiter_catch,
                                       g_object_ref (waiter),
                                       g_object_unref));

  return foundry_jsonrpc_waiter_await (waiter);
}

/**
 * foundry_jsonrpc_driver_notify:
 * @self: a [class@Foundry.JsonrpcDriver]
 *
 * Returns: (transfer full): a future that resolves to any value once
 *   the message has been queued for delivery
 */
DexFuture *
foundry_jsonrpc_driver_notify (FoundryJsonrpcDriver *self,
                               const char           *method,
                               JsonNode             *params)
{
  g_autoptr(FoundryJsonrpcWaiter) waiter = NULL;
  g_autoptr(JsonObject) object = NULL;
  g_autoptr(JsonNode) node = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_JSONRPC_DRIVER (self));
  dex_return_error_if_fail (method != NULL);

  object = json_object_new ();

  json_object_set_string_member (object, "jsonrpc", "2.0");
  json_object_set_string_member (object, "method", method);

  if (params != NULL)
    json_object_set_member (object, "params", json_node_ref (params));
  else
    json_object_set_null_member (object, "params");

  node = json_node_new (JSON_NODE_OBJECT);
  json_node_set_object (node, object);

  waiter = foundry_jsonrpc_waiter_new (node, NULL);

  return dex_future_catch (dex_channel_send (self->output_channel,
                                             dex_future_new_take_object (g_object_ref (waiter))),
                           foundry_jsonrpc_waiter_catch,
                           g_object_ref (waiter),
                           g_object_unref);
}

/**
 * foundry_jsonrpc_driver_reply_with_error:
 * @self: a [class@Foundry.JsonrpcDriver]
 *
 * Returns: (transfer full): a future that resolves to any value once
 *   the message has been queued for delivery
 */
DexFuture *
foundry_jsonrpc_driver_reply_with_error (FoundryJsonrpcDriver *self,
                                         JsonNode             *id,
                                         int                   code,
                                         const char           *message)
{
  g_autoptr(FoundryJsonrpcWaiter) waiter = NULL;
  g_autoptr(JsonObject) object = NULL;
  g_autoptr(JsonObject) error = NULL;
  g_autoptr(JsonNode) node = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_JSONRPC_DRIVER (self));

  object = json_object_new ();
  error = json_object_new ();

  json_object_set_int_member (error, "code", code);
  json_object_set_string_member (error, "message", message);

  json_object_set_string_member (object, "jsonrpc", "2.0");
  json_object_set_member (object, "id", json_node_ref (id));
  json_object_set_object_member (object, "error", json_object_ref (error));

  node = json_node_new (JSON_NODE_OBJECT);
  json_node_set_object (node, object);

  waiter = foundry_jsonrpc_waiter_new (node, NULL);

  return dex_future_catch (dex_channel_send (self->output_channel,
                                             dex_future_new_take_object (g_object_ref (waiter))),
                           foundry_jsonrpc_waiter_catch,
                           g_object_ref (waiter),
                           g_object_unref);
}

typedef struct _Worker
{
  GWeakRef                 self_wr;
  DexChannel              *output_channel;
  FoundryJsonOutputStream *output;
  FoundryJsonInputStream  *input;
  GBytes                  *delimiter;
  FoundryJsonrpcStyle      style : 2;
} Worker;

static void
worker_free (Worker *state)
{
  g_weak_ref_clear (&state->self_wr);
  dex_clear (&state->output_channel);
  g_clear_object (&state->output);
  g_clear_object (&state->input);
  g_clear_pointer (&state->delimiter, g_bytes_unref);
  g_free (state);
}

static DexFuture *
foundry_jsonrpc_driver_read (FoundryJsonrpcStyle     style,
                             FoundryJsonInputStream *stream,
                             GBytes                 *delimiter)
{
  const char *data = NULL;
  gsize size = 0;

  if (style == FOUNDRY_JSONRPC_STYLE_HTTP)
    return foundry_json_input_stream_read_http (stream);

  if (delimiter)
    data = g_bytes_get_data (delimiter, &size);

  return foundry_json_input_stream_read_upto (stream, data, size);
}

static DexFuture *
foundry_jsonrpc_driver_worker (gpointer data)
{
  Worker *state = data;
  g_autoptr(DexFuture) next_read = NULL;
  g_autoptr(DexFuture) next_write = NULL;

  g_assert (state != NULL);
  g_assert (state->output_channel != NULL);
  g_assert (state->output != NULL);
  g_assert (state->input != NULL);

  for (;;)
    {
      g_autoptr(FoundryJsonrpcDriver) self = NULL;
      g_autoptr(GError) error = NULL;

      if (next_read == NULL)
        {
          next_read = foundry_jsonrpc_driver_read (state->style, state->input, state->delimiter);
          dex_future_disown (dex_ref (next_read));
        }

      if (next_write == NULL)
        {
          next_write = dex_channel_receive (state->output_channel);
          dex_future_disown (dex_ref (next_write));
        }

      /* Wait until there is something to read or write */
      if (dex_await (dex_future_any (dex_ref (next_read),
                                     dex_ref (next_write),
                                     NULL),
                     NULL))
        {
          /* If we read a message, get the bytes and decode it for
           * delivering to the application.
           */
          if (dex_future_is_resolved (next_read))
            {
              g_autoptr(JsonNode) node = NULL;

              if (!(node = dex_await_boxed (g_steal_pointer (&next_read), &error)))
                return dex_future_new_for_error (g_steal_pointer (&error));

              if ((self = g_weak_ref_get (&state->self_wr)))
                {
                  foundry_jsonrpc_driver_handle_message (self, node);
                  g_clear_object (&self);
                }
            }

          /* If we got a message to write, then submit it now. This
           * awaits for the message to be buffered because otherwise
           * we could end up in a situation where we try to submit
           * two outgoing messages at the same time.
           */
          if (dex_future_is_resolved (next_write))
            {
              g_autoptr(FoundryJsonrpcWaiter) waiter = dex_await_object (g_steal_pointer (&next_write), NULL);

              g_assert (!waiter || FOUNDRY_IS_JSONRPC_WAITER (waiter));

              if (waiter != NULL)
                {
                  JsonNode *node = foundry_jsonrpc_waiter_get_node (waiter);
                  GHashTable *headers;

                  if (state->style == FOUNDRY_JSONRPC_STYLE_HTTP)
                    headers = empty_headers;
                  else
                    headers = NULL;

                  if (!dex_await (foundry_json_output_stream_write (state->output, headers, node, state->delimiter), &error))
                    return dex_future_new_for_error (g_steal_pointer (&error));
                }
            }
        }

      g_assert (self == NULL);

      /* Before we try to run again, make sure that our client
       * has not been disposed. If so, then we can just bail.
       */
      if (!(self = g_weak_ref_get (&state->self_wr)))
        break;
    }

  return dex_future_new_true ();
}

static DexFuture *
foundry_jsonrpc_driver_panic (DexFuture *completed,
                              gpointer   user_data)
{
  GWeakRef *wr = user_data;
  g_autoptr(FoundryJsonrpcDriver) self = g_weak_ref_get (wr);

  g_assert (!self || FOUNDRY_IS_JSONRPC_DRIVER (self));

  if (self != NULL)
    foundry_jsonrpc_driver_stop (self);

  return dex_future_new_true ();
}

void
foundry_jsonrpc_driver_start (FoundryJsonrpcDriver *self)
{
  Worker *state;

  g_return_if_fail (FOUNDRY_IS_JSONRPC_DRIVER (self));
  g_return_if_fail (G_IS_IO_STREAM (self->stream));
  g_return_if_fail (self->input != NULL);
  g_return_if_fail (self->output != NULL);

  state = g_new0 (Worker, 1);
  g_weak_ref_init (&state->self_wr, self);
  state->input = g_object_ref (self->input);
  state->output = g_object_ref (self->output);
  state->output_channel = dex_ref (self->output_channel);
  state->delimiter = self->delimiter ? g_bytes_ref (self->delimiter) : NULL;
  state->style = self->style;

  dex_future_disown (dex_future_finally (dex_scheduler_spawn (NULL, 0,
                                                              foundry_jsonrpc_driver_worker,
                                                              state,
                                                              (GDestroyNotify) worker_free),
                                         foundry_jsonrpc_driver_panic,
                                         foundry_weak_ref_new (self),
                                         (GDestroyNotify) foundry_weak_ref_free));

}

void
foundry_jsonrpc_driver_stop (FoundryJsonrpcDriver *self)
{
  g_return_if_fail (FOUNDRY_IS_JSONRPC_DRIVER (self));

  if (self->stream != NULL)
    g_io_stream_close_async (self->stream, 0, NULL, NULL, NULL);

  if (self->requests != NULL)
    {
      GHashTableIter iter;
      gpointer k, v;

      g_hash_table_iter_init (&iter, self->requests);

      while (g_hash_table_iter_next (&iter, &k, &v))
        {
          g_autoptr(JsonNode) stolen_key = k;
          g_autoptr(FoundryJsonrpcWaiter) waiter = v;

          g_hash_table_iter_steal (&iter);

          foundry_jsonrpc_waiter_reject (waiter,
                                         g_error_new_literal (G_IO_ERROR,
                                                              G_IO_ERROR_CLOSED,
                                                              "Connection closed"));
        }

    }
}
