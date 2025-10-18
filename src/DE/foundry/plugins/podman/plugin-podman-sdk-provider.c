/* plugin-podman-sdk-provider.c
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

#include <stdio.h>

#include <json-glib/json-glib.h>
#include <libdex.h>

#include "plugin-distrobox-sdk.h"
#include "plugin-podman-sdk-provider.h"
#include "plugin-podman-sdk.h"
#include "plugin-toolbox-sdk.h"

#define PODMAN_RELOAD_DELAY_SECONDS 3

typedef struct _LabelToType
{
  const char *label;
  const char *value;
  GType type;
} LabelToType;

struct _PluginPodmanSdkProvider
{
  FoundrySdkProvider  parent_instance;
  GFileMonitor       *storage_monitor;
  GFileMonitor       *monitor;
  GArray             *label_to_type;
  guint               queued_update;
};

G_DEFINE_FINAL_TYPE (PluginPodmanSdkProvider, plugin_podman_sdk_provider, FOUNDRY_TYPE_SDK_PROVIDER)

static void
plugin_podman_sdk_provider_set_type_for_label (PluginPodmanSdkProvider *self,
                                               const char              *key,
                                               const char              *value,
                                               GType                    container_type)
{
  LabelToType map;

  g_return_if_fail (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));
  g_return_if_fail (key != NULL);
  g_return_if_fail (g_type_is_a (container_type, PLUGIN_TYPE_PODMAN_SDK));

  map.label = g_intern_string (key);
  map.value = g_intern_string (value);
  map.type = container_type;

  g_array_append_val (self->label_to_type, map);
}

static void
plugin_podman_sdk_provider_storage_dir_changed (PluginPodmanSdkProvider *self,
                                                GFile                   *file,
                                                GFile                   *other_file,
                                                GFileMonitorEvent        event,
                                                GFileMonitor            *monitor)
{
  g_autofree char *name = NULL;

  g_assert (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));
  g_assert (G_IS_FILE (file));
  g_assert (G_IS_FILE_MONITOR (monitor));

  name = g_file_get_basename (file);

  if (g_strcmp0 (name, "db.sql") == 0)
    plugin_podman_sdk_provider_queue_update (self);
}

static DexFuture *
plugin_podman_sdk_provider_load_fiber (gpointer user_data)
{
  PluginPodmanSdkProvider *self = user_data;
  g_autoptr(GFile) file = NULL;
  g_autoptr(GFile) storage_dir = NULL;
  g_autofree char *data_dir = NULL;
  g_autofree char *parent_dir = NULL;

  g_assert (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));

  g_set_str (&data_dir, g_get_user_data_dir ());
  if (data_dir == NULL)
    data_dir = g_build_filename (g_get_home_dir (), ".local", "share", NULL);

  g_assert (data_dir != NULL);

  storage_dir = g_file_new_build_filename (data_dir, "containers", "storage", NULL);
  parent_dir = g_build_filename (data_dir, "containers", "storage", "overlay-containers", NULL);
  file = g_file_new_build_filename (parent_dir, "containers.json", NULL);

  /* If our parent directory does not exist, we won't be able to monitor
   * for changes to the podman json file. Just create it upfront in the
   * same form that it'd be created by podman (mode 0700).
   */
  g_mkdir_with_parents (parent_dir, 0700);

  /* We have two files to monitor for potential updates. The containers.json
   * file is primarily how we've done it. But if we have hopes to track the
   * creation of containers via quadlet, we must monitor db.sql for changes.
   *
   * Since db.sql might not exist if we're the first to set things up, we
   * track changes to the @storage_dir directory. We could filter on what
   * files are changed there, but it isn't frequently changed and we delay
   * three seconds, so that doesn't seem necessary.
   */

  if ((self->monitor = g_file_monitor (file, G_FILE_MONITOR_NONE, NULL, NULL)))
    g_signal_connect_object (self->monitor,
                             "changed",
                             G_CALLBACK (plugin_podman_sdk_provider_queue_update),
                             self,
                             G_CONNECT_SWAPPED);

  if ((self->storage_monitor = g_file_monitor_directory (storage_dir, G_FILE_MONITOR_NONE, NULL, NULL)))
    g_signal_connect_object (self->storage_monitor,
                             "changed",
                             G_CALLBACK (plugin_podman_sdk_provider_storage_dir_changed),
                             self,
                             G_CONNECT_SWAPPED);

  return plugin_podman_sdk_provider_update (self);
}

static DexFuture *
plugin_podman_sdk_provider_load (FoundrySdkProvider *sdk_provider)
{
  PluginPodmanSdkProvider *self = (PluginPodmanSdkProvider *)sdk_provider;

  g_assert (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_podman_sdk_provider_load_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
plugin_podman_sdk_provider_unload (FoundrySdkProvider *sdk_provider)
{
  PluginPodmanSdkProvider *self = (PluginPodmanSdkProvider *)sdk_provider;

  g_assert (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));

  g_clear_object (&self->monitor);
  g_clear_object (&self->storage_monitor);
  g_clear_handle_id (&self->queued_update, g_source_remove);

  return FOUNDRY_SDK_PROVIDER_CLASS (plugin_podman_sdk_provider_parent_class)->unload (sdk_provider);
}

static void
plugin_podman_sdk_provider_finalize (GObject *object)
{
  PluginPodmanSdkProvider *self = (PluginPodmanSdkProvider *)object;

  g_clear_pointer (&self->label_to_type, g_array_unref);

  G_OBJECT_CLASS (plugin_podman_sdk_provider_parent_class)->finalize (object);
}

static void
plugin_podman_sdk_provider_class_init (PluginPodmanSdkProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundrySdkProviderClass *sdk_provider_class = FOUNDRY_SDK_PROVIDER_CLASS (klass);

  object_class->finalize = plugin_podman_sdk_provider_finalize;

  sdk_provider_class->load = plugin_podman_sdk_provider_load;
  sdk_provider_class->unload = plugin_podman_sdk_provider_unload;
}

static void
plugin_podman_sdk_provider_init (PluginPodmanSdkProvider *self)
{
  self->label_to_type = g_array_new (FALSE, FALSE, sizeof (LabelToType));

  /* Prioritize "manager":"distrobox" above toolbox because it erroniously
   * can add com.github.containers.toolbox too! See chergert/ptyxis#245
   * for details.
   */
  plugin_podman_sdk_provider_set_type_for_label (self,
                                                 "manager",
                                                 "distrobox",
                                                 PLUGIN_TYPE_DISTROBOX_SDK);
  plugin_podman_sdk_provider_set_type_for_label (self,
                                                 "com.github.containers.toolbox",
                                                 NULL,
                                                 PLUGIN_TYPE_TOOLBOX_SDK);
}

static gboolean
container_is_infra (JsonObject *object)
{
  JsonNode *is_infra;
  g_assert (object != NULL);

  return json_object_has_member (object, "IsInfra") &&
      (is_infra = json_object_get_member (object, "IsInfra")) &&
      json_node_get_value_type (is_infra) == G_TYPE_BOOLEAN &&
      json_node_get_boolean (is_infra);
}

static gboolean
label_matches (JsonNode          *node,
               const LabelToType *l_to_t)
{
  if (l_to_t->value != NULL)
    return JSON_NODE_HOLDS_VALUE (node) &&
           json_node_get_value_type (node) == G_TYPE_STRING &&
           g_strcmp0 (l_to_t->value, json_node_get_string (node)) == 0;

  return TRUE;
}

static PluginPodmanSdk *
plugin_podman_sdk_provider_deserialize (PluginPodmanSdkProvider *self,
                                        JsonObject              *object)
{
  g_autoptr(PluginPodmanSdk) container = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GError) error = NULL;
  JsonObject *labels_object;
  JsonNode *labels;
  GType gtype;

  g_assert (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));
  g_assert (object != NULL);

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    return NULL;

  gtype = PLUGIN_TYPE_PODMAN_SDK;

  if (json_object_has_member (object, "Labels") &&
      (labels = json_object_get_member (object, "Labels")) &&
      JSON_NODE_HOLDS_OBJECT (labels) &&
      (labels_object = json_node_get_object (labels)))
    {
      for (guint i = 0; i < self->label_to_type->len; i++)
        {
          const LabelToType *l_to_t = &g_array_index (self->label_to_type, LabelToType, i);

          if (json_object_has_member (labels_object, l_to_t->label))
            {
              JsonNode *match = json_object_get_member (labels_object, l_to_t->label);

              if (label_matches (match, l_to_t))
                {
                  gtype = l_to_t->type;
                  break;
                }
            }
        }
    }

  container = g_object_new (gtype,
                            "context", context,
                            NULL);

  if (!plugin_podman_sdk_deserialize (container, object, &error))
    {
      g_warning ("Failed to deserialize container JSON: %s", error->message);
      return NULL;
    }

  return g_steal_pointer (&container);
}

static DexFuture *
plugin_podman_sdk_provider_update_cb (DexFuture *completed,
                                      gpointer   user_data)
{
  PluginPodmanSdkProvider *self = user_data;
  g_autoptr(JsonParser) parser = NULL;
  g_autoptr(GPtrArray) sdks = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *stdout_buf = NULL;
  JsonArray *root_array;
  JsonNode *root;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));

  parser = json_parser_new ();

  if (!(stdout_buf = dex_await_string (dex_ref (completed), &error)) ||
      !json_parser_load_from_data (parser, stdout_buf, -1, &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  sdks = g_ptr_array_new_with_free_func (g_object_unref);

  if ((root = json_parser_get_root (parser)) &&
      JSON_NODE_HOLDS_ARRAY (root) &&
      (root_array = json_node_get_array (root)))
    {
      guint n_elements = json_array_get_length (root_array);

      for (guint i = 0; i < n_elements; i++)
        {
          g_autoptr(PluginPodmanSdk) container = NULL;
          JsonNode *element = json_array_get_element (root_array, i);
          JsonObject *element_object;

          if (JSON_NODE_HOLDS_OBJECT (element) &&
              (element_object = json_node_get_object (element)) &&
              !container_is_infra (element_object) &&
              (container = plugin_podman_sdk_provider_deserialize (self, element_object)))
            g_ptr_array_add (sdks, g_steal_pointer (&container));
        }
    }

  foundry_sdk_provider_merge (FOUNDRY_SDK_PROVIDER (self), sdks);

  return dex_future_new_true ();
}

DexFuture *
plugin_podman_sdk_provider_update (PluginPodmanSdkProvider *self)
{
  const GSubprocessFlags flags = G_SUBPROCESS_FLAGS_STDERR_SILENCE|G_SUBPROCESS_FLAGS_STDOUT_PIPE;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  DexFuture *future;

  g_assert (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));

  launcher = foundry_process_launcher_new ();

  foundry_process_launcher_push_host (launcher);

  foundry_process_launcher_append_argv (launcher, "podman");
  foundry_process_launcher_append_argv (launcher, "ps");
  foundry_process_launcher_append_argv (launcher, "--all");
  foundry_process_launcher_append_argv (launcher, "--format=json");

  /* Ignore failures if podman fails to launch */
  if (!(subprocess = foundry_process_launcher_spawn_with_flags (launcher, flags, &error)))
    return dex_future_new_true ();

  future = foundry_subprocess_communicate_utf8 (subprocess, NULL);
  future = dex_future_then (future,
                            plugin_podman_sdk_provider_update_cb,
                            g_object_ref (self),
                            g_object_unref);

  return future;
}

static gboolean
plugin_podman_sdk_provider_update_source_func (gpointer user_data)
{
  PluginPodmanSdkProvider *self = user_data;

  g_assert (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));

  self->queued_update = 0;
  dex_future_disown (plugin_podman_sdk_provider_update (self));
  return G_SOURCE_REMOVE;
}

void
plugin_podman_sdk_provider_queue_update (PluginPodmanSdkProvider *self)
{
  g_return_if_fail (PLUGIN_IS_PODMAN_SDK_PROVIDER (self));

  if (self->queued_update == 0)
    self->queued_update = g_timeout_add_seconds_full (G_PRIORITY_LOW,
                                                      PODMAN_RELOAD_DELAY_SECONDS,
                                                      plugin_podman_sdk_provider_update_source_func,
                                                      self, NULL);
}

static DexFuture *
plugin_podman_sdk_provider_get_version_fiber (gpointer user_data)
{
  const GSubprocessFlags flags = G_SUBPROCESS_FLAGS_STDERR_SILENCE|G_SUBPROCESS_FLAGS_STDOUT_PIPE;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(JsonParser) parser = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *stdout_buf = NULL;
  JsonObject *obj;
  JsonNode *node;

  launcher = foundry_process_launcher_new ();

  foundry_process_launcher_push_host (launcher);

  foundry_process_launcher_append_argv (launcher, "podman");
  foundry_process_launcher_append_argv (launcher, "version");
  foundry_process_launcher_append_argv (launcher, "--format=json");

  if (!(subprocess = foundry_process_launcher_spawn_with_flags (launcher, flags, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(stdout_buf = dex_await_string (foundry_subprocess_communicate_utf8 (subprocess, NULL), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  parser = json_parser_new ();
  if (!json_parser_load_from_data (parser, stdout_buf, -1, &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if ((node = json_parser_get_root (parser)) &&
      JSON_NODE_HOLDS_OBJECT (node) &&
      (obj = json_node_get_object (node)) &&
      json_object_has_member (obj, "Client") &&
      (node = json_object_get_member (obj, "Client")) &&
      JSON_NODE_HOLDS_OBJECT (node) &&
      (obj = json_node_get_object (node)) &&
      json_object_has_member (obj, "Version") &&
      (node = json_object_get_member (obj, "Version")) &&
      JSON_NODE_HOLDS_VALUE (node))
    return dex_future_new_take_string (g_strdup (json_node_get_string (node)));

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_INVALID_DATA,
                                "Unknown JSON format");
}

static DexFuture *
plugin_podman_sdk_provider_get_version (void)
{
  return dex_scheduler_spawn (NULL, 0,
                              plugin_podman_sdk_provider_get_version_fiber,
                              NULL, NULL);
}

typedef struct
{
  guint major;
  guint minor;
  guint micro;
} Version;

static DexFuture *
plugin_podman_sdk_provider_check_version_cb (DexFuture *completed,
                                             gpointer   user_data)
{
  g_autoptr(GError) error = NULL;
  g_autofree char *version = NULL;
  Version *v = user_data;
  guint pmaj, pmin, pmic;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (DEX_IS_FUTURE (completed));
  g_assert (v != NULL);

  if (!(version = dex_await_string (dex_ref (completed), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (sscanf (version, "%u.%u.%u", &pmaj, &pmin, &pmic) != 3)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_INVALID_DATA,
                                  "Invalid data returned from podman");

  if ((pmaj > v->major) ||
      ((pmaj == v->major) && (pmin > v->minor)) ||
      ((pmaj == v->major) && (pmin == v->minor) && (pmic >= v->micro)))
    return dex_future_new_true ();

  return dex_future_new_false ();
}

DexFuture *
plugin_podman_sdk_provider_check_version (guint major,
                                          guint minor,
                                          guint micro)
{
  Version version = {major, minor, micro};

  return dex_future_then (plugin_podman_sdk_provider_get_version (),
                          plugin_podman_sdk_provider_check_version_cb,
                          g_memdup2 (&version, sizeof version),
                          g_free);
}
