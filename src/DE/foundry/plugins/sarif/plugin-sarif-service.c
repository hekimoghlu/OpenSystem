/* plugin-sarif-service.c
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

#include "foundry-json-input-stream-private.h"

#include "plugin-sarif-diagnostic.h"
#include "plugin-sarif-service.h"

struct _PluginSarifService
{
  FoundryService  parent_instance;
  char           *socket_path;
  DexCancellable *cancellable;
  GListStore     *diagnostics;
  DexFuture      *run_fiber;
  char           *builddir;
};

G_DEFINE_FINAL_TYPE (PluginSarifService, plugin_sarif_service, FOUNDRY_TYPE_SERVICE)

static void
plugin_sarif_service_handle_result (PluginSarifService *self,
                                    JsonNode           *result)
{
  g_autoptr(FoundryDiagnostic) diagnostic = NULL;
  g_autoptr(FoundryContext) context = NULL;

  g_assert (PLUGIN_IS_SARIF_SERVICE (self));

  if (self->diagnostics == NULL ||
      !(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    return;

  if ((diagnostic = plugin_sarif_diagnostic_new (context, result, self->builddir)))
    g_list_store_append (self->diagnostics, diagnostic);
}

static void
plugin_sarif_service_handle_message (PluginSarifService *self,
                                     JsonNode           *node)
{
  const char *method = NULL;
  JsonNode *params = NULL;

  g_assert (PLUGIN_IS_SARIF_SERVICE (self));
  g_assert (node != NULL);

  if (!FOUNDRY_JSON_OBJECT_PARSE (node,
                                  "jsonrpc", "2.0",
                                  "method", FOUNDRY_JSON_NODE_GET_STRING (&method),
                                  "params", FOUNDRY_JSON_NODE_GET_NODE (&params)))
    return;

  if (g_strcmp0 (method, "OnSarifResult") == 0)
    {
      JsonNode *result = NULL;

      if (FOUNDRY_JSON_OBJECT_PARSE (params, "result", FOUNDRY_JSON_NODE_GET_NODE (&result)))
        plugin_sarif_service_handle_result (self, result);
    }
}

typedef struct _Worker
{
  GWeakRef         service_wr;
  GSocketListener *listener;
  DexCancellable  *cancellable;
  char            *path;
} Worker;

static void
worker_finalize (gpointer data)
{
  Worker *state = data;
  g_socket_listener_close (state->listener);
  g_weak_ref_clear (&state->service_wr);
  g_clear_object (&state->listener);
  dex_clear (&state->cancellable);
  g_clear_pointer (&state->path, g_free);
}

static Worker *
worker_ref (Worker *worker)
{
  return g_atomic_rc_box_acquire (worker);
}

static void
worker_unref (Worker *worker)
{
  g_atomic_rc_box_release_full (worker, worker_finalize);
}

static DexFuture *
plugin_sarif_service_drain_fiber (gpointer data)
{
  GSocketConnection *conn = data;
  g_autoptr(FoundryJsonInputStream) stream = NULL;
  GInputStream *base_stream;
  Worker *state;

  g_assert (G_IS_SOCKET_CONNECTION (conn));

  state = g_object_get_data (G_OBJECT (conn), "SARIF_WORKER");

  g_assert (state != NULL);
  g_assert (G_IS_SOCKET_LISTENER (state->listener));
  g_assert (DEX_IS_CANCELLABLE (state->cancellable));

  base_stream = g_io_stream_get_input_stream (G_IO_STREAM (conn));
  stream = foundry_json_input_stream_new (base_stream, FALSE);

  for (;;)
    {
      g_autoptr(PluginSarifService) self = NULL;
      g_autoptr(JsonNode) node = NULL;
      g_autoptr(GError) error = NULL;

      if (!(node = dex_await_boxed (dex_future_first (dex_ref (state->cancellable),
                                                      foundry_json_input_stream_read_http (stream),
                                                      NULL),
                                    &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      if (!(self = g_weak_ref_get (&state->service_wr)))
        break;

      plugin_sarif_service_handle_message (self, node);
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_sarif_service_worker_fiber (gpointer data)
{
  Worker *state = data;

  g_assert (state != NULL);
  g_assert (G_IS_SOCKET_LISTENER (state->listener));
  g_assert (DEX_IS_CANCELLABLE (state->cancellable));

  for (;;)
    {
      g_autoptr(GSocketConnection) conn = NULL;
      g_autoptr(GError) error = NULL;

      if (!(conn = dex_await_object (dex_future_first (dex_ref (state->cancellable),
                                                       dex_socket_listener_accept (state->listener),
                                                       NULL),
                                     &error)))
        {
          if (!g_error_matches (error, G_IO_ERROR, G_IO_ERROR_CANCELLED))
            g_warning ("Failed to accept socket: %s", error->message);
          break;
        }

      g_assert (g_socket_connection_is_connected (conn));

      g_debug ("Accepted `%s` socket on `%s`", G_OBJECT_TYPE_NAME (conn), state->path);

      g_object_set_data_full (G_OBJECT (conn),
                              "SARIF_WORKER",
                              worker_ref (state),
                              (GDestroyNotify) worker_unref);

      dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                              plugin_sarif_service_drain_fiber,
                                              g_object_ref (conn),
                                              g_object_unref));
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_sarif_service_run_fiber (gpointer data)
{
  PluginSarifService *self = data;
  g_autoptr(GSocketListener) listener = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GSocketAddress) address = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *socket_path = NULL;
  g_autofree char *socket_dir = NULL;
  g_autofree char *guid = NULL;
  Worker *state;

  g_assert (PLUGIN_IS_SARIF_SERVICE (self));

  guid = g_dbus_generate_guid ();
  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  socket_path = foundry_context_tmp_filename (context, "sarif", guid, NULL);
  socket_dir = g_path_get_dirname (socket_path);
  address = g_unix_socket_address_new (socket_path);
  listener = g_socket_listener_new ();

  if (!dex_await (dex_mkdir_with_parents (socket_dir, 0750), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!g_socket_listener_add_address (listener,
                                      address,
                                      G_SOCKET_TYPE_STREAM,
                                      G_SOCKET_PROTOCOL_DEFAULT,
                                      NULL, NULL, &error))
    {
      g_debug ("Failed to listen on SARIF socket: %s", error->message);
      return dex_future_new_for_error (g_steal_pointer (&error));
    }

  self->socket_path = g_strdup (socket_path);

  state = g_atomic_rc_box_new0 (Worker);
  state->listener = g_object_ref (listener);
  state->path = g_strdup (socket_path);
  state->cancellable = dex_ref (self->cancellable);
  g_weak_ref_init (&state->service_wr, self);

  dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                          plugin_sarif_service_worker_fiber,
                                          state,
                                          (GDestroyNotify) worker_unref));

  return dex_future_new_take_string (g_steal_pointer (&socket_path));
}

static DexFuture *
plugin_sarif_service_stop (FoundryService *service)
{
  PluginSarifService *self = (PluginSarifService *)service;
  g_autofree char *socket_path = NULL;

  g_assert (PLUGIN_IS_SARIF_SERVICE (self));

  dex_cancellable_cancel (self->cancellable);

  if ((socket_path = g_steal_pointer (&self->socket_path)))
    return dex_unlink (socket_path);

  return dex_future_new_true ();
}

static void
plugin_sarif_service_dispose (GObject *object)
{
  PluginSarifService *self = (PluginSarifService *)object;

  g_clear_pointer (&self->builddir, g_free);
  g_clear_pointer (&self->socket_path, g_free);
  g_clear_object (&self->diagnostics);
  dex_clear (&self->cancellable);
  dex_clear (&self->run_fiber);

  G_OBJECT_CLASS (plugin_sarif_service_parent_class)->dispose (object);
}

static void
plugin_sarif_service_class_init (PluginSarifServiceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  object_class->dispose = plugin_sarif_service_dispose;

  service_class->stop = plugin_sarif_service_stop;
}

static void
plugin_sarif_service_init (PluginSarifService *self)
{
  self->cancellable = dex_cancellable_new ();
  self->diagnostics = g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC);
}

void
plugin_sarif_service_reset (PluginSarifService *self)
{
  g_return_if_fail (PLUGIN_IS_SARIF_SERVICE (self));

  g_list_store_remove_all (self->diagnostics);
}

/**
 * plugin_sarif_service_socket_path:
 * @self: a [class@Plugin.SarifService]
 *
 * Ensures the socket listener is setup and provides the address to the
 * UNIX domain socket.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a string
 *   containing the path, or rejects with error.
 */
DexFuture *
plugin_sarif_service_socket_path (PluginSarifService *self)
{
  g_return_val_if_fail (PLUGIN_IS_SARIF_SERVICE (self), NULL);

  if (self->run_fiber == NULL)
    {
      self->run_fiber = dex_scheduler_spawn (NULL, 0,
                                             plugin_sarif_service_run_fiber,
                                             g_object_ref (self),
                                             g_object_unref);
      dex_future_disown (dex_ref (self->run_fiber));
    }

  return dex_ref (self->run_fiber);
}

/**
 * plugin_sarif_service_list_diagnostics:
 * @self: a [class@Plugin.SarifService]
 *
 * Returns: (transfer full):
 */
GListModel *
plugin_sarif_service_list_diagnostics (PluginSarifService *self)
{
  g_return_val_if_fail (PLUGIN_IS_SARIF_SERVICE (self), NULL);

  return g_object_ref (G_LIST_MODEL (self->diagnostics));
}

void
plugin_sarif_service_set_builddir (PluginSarifService *self,
                                   const char         *builddir)
{
  g_return_if_fail (PLUGIN_IS_SARIF_SERVICE (self));

  g_set_str (&self->builddir, builddir);
}
