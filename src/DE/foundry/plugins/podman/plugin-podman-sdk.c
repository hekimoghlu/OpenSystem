/* plugin-podman-sdk.c
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

#include "plugin-distrobox-sdk.h"
#include "plugin-toolbox-sdk.h"
#include "plugin-podman-sdk.h"

typedef struct
{
  GHashTable *labels;
  DexPromise *started;
} PluginPodmanSdkPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (PluginPodmanSdk, plugin_podman_sdk, FOUNDRY_TYPE_SDK)

static void
maybe_start_cb (GObject      *object,
                GAsyncResult *result,
                gpointer      user_data)
{
  GSubprocess *subprocess = (GSubprocess *)object;
  g_autoptr(DexPromise) promise = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (G_IS_SUBPROCESS (subprocess));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (DEX_IS_PROMISE (promise));

  if (!g_subprocess_wait_check_finish (subprocess, result, &error))
    dex_promise_reject (promise, g_steal_pointer (&error));
  else
    dex_promise_resolve_boolean (promise, TRUE);
}

static DexFuture *
maybe_start (PluginPodmanSdk *self)
{
  PluginPodmanSdkPrivate *priv = plugin_podman_sdk_get_instance_private (self);
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *id = NULL;

  g_assert (PLUGIN_IS_PODMAN_SDK (self));

  if (priv->started != NULL)
    return dex_ref (DEX_FUTURE (priv->started));

  priv->started = dex_promise_new_cancellable ();

  id = foundry_sdk_dup_id (FOUNDRY_SDK (self));

  /* If this is distrobox, just skip starting as it will start
   * the container manually inside. This fixes an issue where
   * it has a race with the container being started outside
   * of distrobox via podman directly.
   *
   * https://gitlab.gnome.org/chergert/ptyxis/-/issues/31
   */
  if (PLUGIN_IS_DISTROBOX_SDK (self))
    {
      dex_promise_resolve_boolean (priv->started, TRUE);
      return dex_ref (DEX_FUTURE (priv->started));
    }

  launcher = foundry_process_launcher_new ();

  /* In case we're sandboxed */
  foundry_process_launcher_push_host (launcher);

  foundry_process_launcher_append_argv (launcher, "podman");
  foundry_process_launcher_append_argv (launcher, "start");
  foundry_process_launcher_append_argv (launcher, id);
  foundry_process_launcher_take_fd (launcher, -1, STDIN_FILENO);
  foundry_process_launcher_take_fd (launcher, -1, STDOUT_FILENO);
  foundry_process_launcher_take_fd (launcher, -1, STDERR_FILENO);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  g_subprocess_wait_check_async (subprocess,
                                 dex_promise_get_cancellable (priv->started),
                                 maybe_start_cb,
                                 dex_ref (priv->started));

  return dex_ref (DEX_FUTURE (priv->started));
}

static void
foundry_podman_sdk_deserialize_labels (PluginPodmanSdk *self,
                                       JsonObject      *labels)
{
  PluginPodmanSdkPrivate *priv = plugin_podman_sdk_get_instance_private (self);
  JsonObjectIter iter;
  const char *key;
  JsonNode *value;

  g_assert (PLUGIN_IS_PODMAN_SDK (self));
  g_assert (labels != NULL);

  json_object_iter_init (&iter, labels);

  while (json_object_iter_next (&iter, &key, &value))
    {
      if (JSON_NODE_HOLDS_VALUE (value) &&
          json_node_get_value_type (value) == G_TYPE_STRING)
        {
          const char *value_str = json_node_get_string (value);

          g_hash_table_insert (priv->labels, g_strdup (key), g_strdup (value_str));
        }
    }
}

static void
foundry_podman_sdk_deserialize_name (PluginPodmanSdk *self,
                                     JsonArray       *names)
{
  g_assert (PLUGIN_IS_PODMAN_SDK (self));
  g_assert (names != NULL);

  if (json_array_get_length (names) > 0)
    {
      JsonNode *element = json_array_get_element (names, 0);

      if (element != NULL &&
          JSON_NODE_HOLDS_VALUE (element) &&
          json_node_get_value_type (element) == G_TYPE_STRING)
        foundry_sdk_set_name (FOUNDRY_SDK (self),
                              json_node_get_string (element));
    }
}

static gboolean
plugin_podman_sdk_real_deserialize (PluginPodmanSdk  *self,
                                    JsonObject       *object,
                                    GError          **error)
{
  PluginPodmanSdkPrivate *priv = plugin_podman_sdk_get_instance_private (self);
  const char *arch;
  JsonObject *labels_object;
  JsonArray *names_array;
  JsonNode *names;
  JsonNode *labels;
  JsonNode *id;

  g_assert (PLUGIN_IS_PODMAN_SDK (self));
  g_assert (object != NULL);

  if (!(json_object_has_member (object, "Id") &&
        (id = json_object_get_member (object, "Id")) &&
        JSON_NODE_HOLDS_VALUE (id) &&
        json_node_get_value_type (id) == G_TYPE_STRING))
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_INVALID_DATA,
                   "Failed to locate Id in podman container description");
      return FALSE;
    }

  foundry_sdk_set_id (FOUNDRY_SDK (self),
                      json_node_get_string (id));

  if (json_object_has_member (object, "Labels") &&
      (labels = json_object_get_member (object, "Labels")) &&
      JSON_NODE_HOLDS_OBJECT (labels) &&
      (labels_object = json_node_get_object (labels)))
    foundry_podman_sdk_deserialize_labels (self, labels_object);

  if (json_object_has_member (object, "Names") &&
      (names = json_object_get_member (object, "Names")) &&
      JSON_NODE_HOLDS_ARRAY (names) &&
      (names_array = json_node_get_array (names)))
    foundry_podman_sdk_deserialize_name (self, names_array);

  if ((arch = g_hash_table_lookup (priv->labels, "architecture")))
    foundry_sdk_set_arch (FOUNDRY_SDK (self), arch);
  else
    foundry_sdk_set_arch (FOUNDRY_SDK (self), foundry_get_default_arch ());

  return TRUE;
}

typedef struct _Prepare
{
  PluginPodmanSdk           *self;
  FoundryBuildPipeline      *pipeline;
  FoundryProcessLauncher    *launcher;
  FoundryBuildPipelinePhase  phase;
} Prepare;

static void
prepare_finalize (gpointer data)
{
  Prepare *state = data;

  g_clear_object (&state->self);
  g_clear_object (&state->pipeline);
  g_clear_object (&state->launcher);
}

static Prepare *
prepare_ref (Prepare *state)
{
  return g_atomic_rc_box_acquire (state);
}

static void
prepare_unref (Prepare *state)
{
  g_atomic_rc_box_release_full (state, prepare_finalize);
}

static gboolean
plugin_podman_sdk_prepare_cb (FoundryProcessLauncher  *launcher,
                              const char * const      *argv,
                              const char * const      *env,
                              const char              *cwd,
                              FoundryUnixFDMap        *unix_fd_map,
                              gpointer                 user_data,
                              GError                 **error)
{
  Prepare *state = user_data;
  g_autofree char *id = NULL;
  gboolean has_tty = FALSE;
  int max_dest_fd;

  g_assert (state != NULL);
  g_assert (PLUGIN_IS_PODMAN_SDK (state->self));
  g_assert (!state->pipeline || FOUNDRY_IS_BUILD_PIPELINE (state->pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (state->launcher));
  g_assert (state->launcher == launcher);
  g_assert (argv != NULL);
  g_assert (env != NULL);
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (unix_fd_map));

  id = foundry_sdk_dup_id (FOUNDRY_SDK (state->self));

  /* Make sure that we request TTY ioctls if necessary */
  if (foundry_unix_fd_map_stdin_isatty (unix_fd_map) ||
      foundry_unix_fd_map_stdout_isatty (unix_fd_map) ||
      foundry_unix_fd_map_stderr_isatty (unix_fd_map))
    has_tty = TRUE;

  /* Make sure we can pass the FDs down */
  if (!foundry_process_launcher_merge_unix_fd_map (launcher, unix_fd_map, error))
    return FALSE;

  /* Setup basic podman-exec command */
  foundry_process_launcher_append_argv (launcher, "podman");
  foundry_process_launcher_append_argv (launcher, "exec");
  foundry_process_launcher_append_argv (launcher, "--privileged");
  foundry_process_launcher_append_argv (launcher, "--interactive");

  /* Make sure that we request TTY ioctls if necessary */
  if (has_tty)
    foundry_process_launcher_append_argv (launcher, "--tty");

  /* If there is a CWD specified, then apply it. However, podman containers
   * won't necessarily have the user home directory in them except for when
   * using toolbox/distrobox. So only apply in those cases.
   */
  if (PLUGIN_IS_TOOLBOX_SDK (state->self) || PLUGIN_IS_DISTROBOX_SDK (state->self))
    {
      foundry_process_launcher_append_formatted (launcher, "--user=%s", g_get_user_name ());
      if (cwd != NULL)
        foundry_process_launcher_append_formatted (launcher, "--workdir=%s", cwd);
    }

  /* From podman-exec(1):
   *
   * Pass down to the process N additional file descriptors (in addition to
   * 0, 1, 2).  The total FDs will be 3+N.
   */
  if ((max_dest_fd = foundry_unix_fd_map_get_max_dest_fd (unix_fd_map)) > 2)
    foundry_process_launcher_append_formatted (launcher, "--preserve-fds=%d", max_dest_fd-2);

  /* Sspecify --detach-keys to avoid it stealing our ctrl+p.
   *
   * https://github.com/containers/toolbox/issues/394
   */
  foundry_process_launcher_append_argv (launcher, "--detach-keys=");

  /* Append --env=FOO=BAR environment variables */
  for (guint i = 0; env[i]; i++)
    foundry_process_launcher_append_formatted (launcher, "--env=%s", env[i]);

  /* Now specify our runtime identifier */
  foundry_process_launcher_append_argv (launcher, id);

  /* Finally, propagate the upper layer's command arguments */
  foundry_process_launcher_append_args (launcher, argv);

  return TRUE;
}

static DexFuture *
plugin_podman_sdk_prepare_fiber (gpointer user_data)
{
  Prepare *state = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (state != NULL);
  g_assert (PLUGIN_IS_PODMAN_SDK (state->self));
  g_assert (!state->pipeline || FOUNDRY_IS_BUILD_PIPELINE (state->pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (state->launcher));

  /* First make sure our container is started */
  if (!dex_await (maybe_start (state->self), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  foundry_process_launcher_push (state->launcher,
                                 plugin_podman_sdk_prepare_cb,
                                 prepare_ref (state),
                                 (GDestroyNotify) prepare_unref);

  return dex_future_new_true ();
}

static DexFuture *
plugin_podman_sdk_prepare_to_build (FoundrySdk                *sdk,
                                    FoundryBuildPipeline      *pipeline,
                                    FoundryProcessLauncher    *launcher,
                                    FoundryBuildPipelinePhase  phase)
{
  PluginPodmanSdk *self = (PluginPodmanSdk *)sdk;
  Prepare *state;

  g_assert (PLUGIN_IS_PODMAN_SDK (self));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  state = g_atomic_rc_box_new0 (Prepare);
  state->phase = phase;
  g_set_object (&state->self, self);
  g_set_object (&state->pipeline, pipeline);
  g_set_object (&state->launcher, launcher);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_podman_sdk_prepare_fiber,
                              g_steal_pointer (&state),
                              (GDestroyNotify) prepare_unref);
}

static DexFuture *
plugin_podman_sdk_prepare_to_run (FoundrySdk             *sdk,
                                  FoundryBuildPipeline   *pipeline,
                                  FoundryProcessLauncher *launcher)
{
  return plugin_podman_sdk_prepare_to_build (sdk, pipeline, launcher, 0);
}

static void
plugin_podman_sdk_finalize (GObject *object)
{
  PluginPodmanSdk *self = (PluginPodmanSdk *)object;
  PluginPodmanSdkPrivate *priv = plugin_podman_sdk_get_instance_private (self);

  dex_clear (&priv->started);
  g_clear_pointer (&priv->labels, g_hash_table_unref);

  G_OBJECT_CLASS (plugin_podman_sdk_parent_class)->finalize (object);
}

static void
plugin_podman_sdk_class_init (PluginPodmanSdkClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundrySdkClass *sdk_class = FOUNDRY_SDK_CLASS (klass);

  object_class->finalize = plugin_podman_sdk_finalize;

  sdk_class->prepare_to_build = plugin_podman_sdk_prepare_to_build;
  sdk_class->prepare_to_run = plugin_podman_sdk_prepare_to_run;

  klass->deserialize = plugin_podman_sdk_real_deserialize;
}

static void
plugin_podman_sdk_init (PluginPodmanSdk *self)
{
  PluginPodmanSdkPrivate *priv = plugin_podman_sdk_get_instance_private (self);

  priv->labels = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_free);

  foundry_sdk_set_kind (FOUNDRY_SDK (self), "podman");
  foundry_sdk_set_installed (FOUNDRY_SDK (self), TRUE);
}

gboolean
plugin_podman_sdk_deserialize (PluginPodmanSdk  *self,
                               JsonObject       *object,
                               GError          **error)
{
  g_return_val_if_fail (PLUGIN_IS_PODMAN_SDK (self), FALSE);
  g_return_val_if_fail (object != NULL, FALSE);

  return PLUGIN_PODMAN_SDK_GET_CLASS (self)->deserialize (self, object, error);
}
