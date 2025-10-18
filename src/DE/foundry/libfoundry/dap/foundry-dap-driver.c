/* foundry-dap-driver.c
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
#include "foundry-json-node.h"
#include "foundry-json-output-stream-private.h"
#include "foundry-dap-driver-private.h"
#include "foundry-dap-waiter-private.h"
#include "foundry-util-private.h"

struct _FoundryDapDriver
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

G_DEFINE_FINAL_TYPE (FoundryDapDriver, foundry_dap_driver, G_TYPE_OBJECT)

enum {
  SIGNAL_HANDLE_REQUEST,
  SIGNAL_EVENT,
  N_SIGNALS
};

static GParamSpec *properties[N_PROPS];
static guint signals[N_SIGNALS];
static GHashTable *empty_headers;

static void
foundry_dap_driver_handle_message (FoundryDapDriver *self,
                                   JsonNode         *node)
{
  const char *type;
  gint64 seq = 0;

  g_assert (FOUNDRY_IS_DAP_DRIVER (self));
  g_assert (node != NULL);

  if (FOUNDRY_JSON_OBJECT_PARSE (node,
                                 "type", FOUNDRY_JSON_NODE_GET_STRING (&type),
                                 "seq", FOUNDRY_JSON_NODE_GET_INT (&seq)) &&
      g_strv_contains (FOUNDRY_STRV_INIT ("event", "request", "response"), type))
    {
      if (g_strcmp0 (type, "event") == 0)
        {
          g_signal_emit (self, signals[SIGNAL_EVENT], 0, node);
          return;
        }
      else if (g_strcmp0 (type, "request") == 0)
        {
          gboolean ret = FALSE;
          g_signal_emit (self, signals[SIGNAL_HANDLE_REQUEST], 0, node, &ret);
          return;
        }
      else if (g_strcmp0 (type, "response") == 0)
        {
          g_autoptr(FoundryDapWaiter) waiter = NULL;
          gint64 request_seq;

          if (FOUNDRY_JSON_OBJECT_PARSE (node,
                                         "request_seq", FOUNDRY_JSON_NODE_GET_INT (&request_seq)) &&
              g_hash_table_steal_extended (self->requests,
                                           GSIZE_TO_POINTER (request_seq),
                                           NULL,
                                           (gpointer *)&waiter))
            foundry_dap_waiter_reply (waiter, json_node_ref (node));

          return;
        }
    }

  /* Protocol error */
  g_io_stream_close_async (self->stream, 0, NULL, NULL, NULL);
}

static void
foundry_dap_driver_dispose (GObject *object)
{
  FoundryDapDriver *self = (FoundryDapDriver *)object;

  if (self->output_channel)
    {
      dex_channel_close_send (self->output_channel);
      dex_channel_close_receive (self->output_channel);
      dex_clear (&self->output_channel);
    }

  foundry_dap_driver_stop (self);

  g_clear_object (&self->input);
  g_clear_object (&self->output);
  g_clear_object (&self->stream);

  g_clear_pointer (&self->requests, g_hash_table_unref);
  g_clear_pointer (&self->delimiter, g_bytes_unref);

  G_OBJECT_CLASS (foundry_dap_driver_parent_class)->dispose (object);
}

static void
foundry_dap_driver_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryDapDriver *self = FOUNDRY_DAP_DRIVER (object);

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
foundry_dap_driver_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  FoundryDapDriver *self = FOUNDRY_DAP_DRIVER (object);

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
foundry_dap_driver_class_init (FoundryDapDriverClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_dap_driver_dispose;
  object_class->get_property = foundry_dap_driver_get_property;
  object_class->set_property = foundry_dap_driver_set_property;

  properties[PROP_STREAM] =
    g_param_spec_object ("stream", NULL, NULL,
                         G_TYPE_IO_STREAM,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  /**
   * FoundryDapDriver::handle-request:
   * @self: a [class@Foundry.JsonrpcDriver]
   * @method:
   * @params: (nullable):
   * @id:
   *
   * Returns: %TRUE if handled; otherwise %FALSE
   */
  signals[SIGNAL_HANDLE_REQUEST] =
    g_signal_new ("handle-request",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  g_signal_accumulator_true_handled, NULL,
                  NULL,
                  G_TYPE_BOOLEAN, 1, JSON_TYPE_NODE);

  signals[SIGNAL_EVENT] =
    g_signal_new ("event",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 1, JSON_TYPE_NODE);

  empty_headers = g_hash_table_new (NULL, NULL);
}

static void
foundry_dap_driver_init (FoundryDapDriver *self)
{
  self->output_channel = dex_channel_new (0);
  self->requests = g_hash_table_new_full (NULL, NULL, NULL, g_object_unref);
}

FoundryDapDriver *
foundry_dap_driver_new (GIOStream           *stream,
                            FoundryJsonrpcStyle  style)
{
  FoundryDapDriver *self;
  GInputStream *input;
  GOutputStream *output;

  g_return_val_if_fail (G_IS_IO_STREAM (stream), NULL);
  g_return_val_if_fail (style > 0, NULL);
  g_return_val_if_fail (style <= FOUNDRY_JSONRPC_STYLE_NIL, NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_DRIVER,
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
 * foundry_dap_driver_call:
 * @self: a [class@Foundry.JsonrpcDriver]
 *
 * Returns: (transfer full): a future that resolves to a [struct@Json.Node]
 *   containing the reply.
 */
DexFuture *
foundry_dap_driver_call (FoundryDapDriver *self,
                         JsonNode         *params)
{
  g_autoptr(FoundryDapWaiter) waiter = NULL;
  gint64 seq;

  dex_return_error_if_fail (FOUNDRY_IS_DAP_DRIVER (self));
  dex_return_error_if_fail (params != NULL);
  dex_return_error_if_fail (JSON_NODE_HOLDS_OBJECT (params));

  seq = ++self->last_seq;
  json_object_set_int_member (json_node_get_object (params), "seq", seq);
  waiter = foundry_dap_waiter_new (params, seq);

  g_hash_table_replace (self->requests,
                        GSIZE_TO_POINTER (seq),
                        g_object_ref (waiter));

  dex_future_disown (dex_future_catch (dex_channel_send (self->output_channel,
                                                         dex_future_new_take_object (g_object_ref (waiter))),
                                       foundry_dap_waiter_catch,
                                       g_object_ref (waiter),
                                       g_object_unref));

  return foundry_dap_waiter_await (waiter);
}

/**
 * foundry_dap_driver_send:
 * @self: a [class@Foundry.JsonrpcDriver]
 *
 * Returns: (transfer full): a future that resolves to any value once
 *   the message has been queued for delivery
 */
DexFuture *
foundry_dap_driver_send (FoundryDapDriver *self,
                         JsonNode         *params)
{
  g_autoptr(FoundryDapWaiter) waiter = NULL;
  gint64 seq;

  dex_return_error_if_fail (FOUNDRY_IS_DAP_DRIVER (self));
  dex_return_error_if_fail (params != NULL);
  dex_return_error_if_fail (JSON_NODE_HOLDS_OBJECT (params));

  seq = ++self->last_seq;
  json_object_set_int_member (json_node_get_object (params), "seq", seq);
  waiter = foundry_dap_waiter_new (params, seq);

  return dex_future_catch (dex_channel_send (self->output_channel,
                                             dex_future_new_take_object (g_object_ref (waiter))),
                           foundry_dap_waiter_catch,
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
foundry_dap_driver_read (FoundryJsonrpcStyle     style,
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
foundry_dap_driver_worker (gpointer data)
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
      g_autoptr(FoundryDapDriver) self = NULL;
      g_autoptr(GError) error = NULL;

      if (next_read == NULL)
        {
          next_read = foundry_dap_driver_read (state->style, state->input, state->delimiter);
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
                  foundry_dap_driver_handle_message (self, node);
                  g_clear_object (&self);
                }
            }
          else if (dex_future_is_rejected (next_read))
            return dex_ref (next_read);

          /* If we got a message to write, then submit it now. This
           * awaits for the message to be buffered because otherwise
           * we could end up in a situation where we try to submit
           * two outgoing messages at the same time.
           */
          if (dex_future_is_resolved (next_write))
            {
              g_autoptr(FoundryDapWaiter) waiter = dex_await_object (g_steal_pointer (&next_write), NULL);

              g_assert (!waiter || FOUNDRY_IS_DAP_WAITER (waiter));

              if (waiter != NULL)
                {
                  JsonNode *node = foundry_dap_waiter_get_node (waiter);
                  GHashTable *headers;

                  if (state->style == FOUNDRY_JSONRPC_STYLE_HTTP)
                    headers = empty_headers;
                  else
                    headers = NULL;

                  if (!dex_await (foundry_json_output_stream_write (state->output, headers, node, state->delimiter), &error))
                    return dex_future_new_for_error (g_steal_pointer (&error));
                }
            }
          else if (dex_future_is_rejected (next_write))
            return dex_ref (next_write);
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
foundry_dap_driver_panic (DexFuture *completed,
                          gpointer   user_data)
{
  GWeakRef *wr = user_data;
  g_autoptr(FoundryDapDriver) self = g_weak_ref_get (wr);

  g_assert (!self || FOUNDRY_IS_DAP_DRIVER (self));

  if (self != NULL)
    g_debug ("`%s` at %p worker has exited",
             G_OBJECT_TYPE_NAME (self), self);

  if (self != NULL)
    foundry_dap_driver_stop (self);

  return dex_future_new_true ();
}

void
foundry_dap_driver_start (FoundryDapDriver *self)
{
  Worker *state;

  g_return_if_fail (FOUNDRY_IS_DAP_DRIVER (self));
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
                                                              foundry_dap_driver_worker,
                                                              state,
                                                              (GDestroyNotify) worker_free),
                                         foundry_dap_driver_panic,
                                         foundry_weak_ref_new (self),
                                         (GDestroyNotify) foundry_weak_ref_free));

}

void
foundry_dap_driver_stop (FoundryDapDriver *self)
{
  g_return_if_fail (FOUNDRY_IS_DAP_DRIVER (self));

  if (self->stream != NULL)
    g_io_stream_close_async (self->stream, 0, NULL, NULL, NULL);

  if (self->requests != NULL)
    {
      GHashTableIter iter;
      gpointer k, v;

      g_hash_table_iter_init (&iter, self->requests);

      while (g_hash_table_iter_next (&iter, &k, &v))
        {
          g_autoptr(FoundryDapWaiter) waiter = v;

          g_hash_table_iter_steal (&iter);

          foundry_dap_waiter_reject (waiter,
                                     g_error_new_literal (G_IO_ERROR,
                                                          G_IO_ERROR_CLOSED,
                                                          "Connection closed"));
        }

    }
}
