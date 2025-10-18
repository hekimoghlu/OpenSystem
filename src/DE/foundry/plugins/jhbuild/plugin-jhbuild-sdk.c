/* plugin-jhbuild-sdk.c
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

#include "plugin-jhbuild-sdk.h"

struct _PluginJhbuildSdk
{
  FoundrySdk parent_instance;
  char *install_prefix;
  char *library_dir;
};

G_DEFINE_FINAL_TYPE (PluginJhbuildSdk, plugin_jhbuild_sdk, FOUNDRY_TYPE_SDK)

static gboolean
plugin_jhbuild_sdk_prepare_cb (FoundryProcessLauncher  *launcher,
                               const char * const      *argv,
                               const char * const      *env,
                               const char              *cwd,
                               FoundryUnixFDMap        *unix_fd_map,
                               gpointer                 user_data,
                               GError                 **error)
{
  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (unix_fd_map));

  /* First merge our FDs so we can be sure there are no collisions (there
   * shouldn't be because we didn't set anything here).
   */
  if (!foundry_process_launcher_merge_unix_fd_map (launcher, unix_fd_map, error))
    FOUNDRY_RETURN (FALSE);

  /* We always take the CWD of the upper layer */
  foundry_process_launcher_set_cwd (launcher, cwd);

  /* We rewrite the argv to be "jhbuild run ..." */
  foundry_process_launcher_append_argv (launcher, "jhbuild");
  foundry_process_launcher_append_argv (launcher, "run");

  /* If there is an environment to deliver, then we want that passed to the
   * subprocess but not to affect the parent process (jhbuild). So it will now
   * look something like "jhbuild run env FOO=BAR ..."
   */
  if (env != NULL && env[0] != NULL)
    {
      foundry_process_launcher_append_argv (launcher, "env");
      foundry_process_launcher_append_args (launcher, env);
    }

  /* And now we can add the argv of the upper layers so it might look something
   * like "jhbuild run env FOO=BAR valgrind env BAR=BAZ my-program"
   */
  foundry_process_launcher_append_args (launcher, argv);

  FOUNDRY_RETURN (TRUE);
}

static DexFuture *
plugin_jhbuild_sdk_prepare_to_build (FoundrySdk                *sdk,
                                     FoundryBuildPipeline      *pipeline,
                                     FoundryProcessLauncher    *launcher,
                                     FoundryBuildPipelinePhase  phase)
{
  g_assert (PLUGIN_IS_JHBUILD_SDK (sdk));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  foundry_process_launcher_push_host (launcher);
  foundry_process_launcher_add_minimal_environment (launcher);
  foundry_process_launcher_push (launcher,
                                 plugin_jhbuild_sdk_prepare_cb,
                                 NULL, NULL);

  return dex_future_new_true ();
}

static DexFuture *
plugin_jhbuild_sdk_prepare_to_run (FoundrySdk             *sdk,
                                   FoundryBuildPipeline   *pipeline,
                                   FoundryProcessLauncher *launcher)
{
  g_assert (PLUGIN_IS_JHBUILD_SDK (sdk));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  foundry_process_launcher_push_host (launcher);
  foundry_process_launcher_add_minimal_environment (launcher);
  foundry_process_launcher_push (launcher,
                                 plugin_jhbuild_sdk_prepare_cb,
                                 NULL, NULL);

  return dex_future_new_true ();
}

static char *
plugin_jhbuild_sdk_dup_config_option (FoundrySdk             *sdk,
                                      FoundrySdkConfigOption  option)
{
  PluginJhbuildSdk *self = PLUGIN_JHBUILD_SDK (sdk);

  switch (option)
    {
    case FOUNDRY_SDK_CONFIG_OPTION_PREFIX:
      return g_strdup (self->install_prefix);

    case FOUNDRY_SDK_CONFIG_OPTION_LIBDIR:
      return g_strdup (self->library_dir);

    default:
      return NULL;
    }
}

static void
plugin_jhbuild_sdk_finalize (GObject *object)
{
  PluginJhbuildSdk *self = PLUGIN_JHBUILD_SDK (object);

  g_clear_pointer (&self->library_dir, g_free);
  g_clear_pointer (&self->install_prefix, g_free);

  G_OBJECT_CLASS (plugin_jhbuild_sdk_parent_class)->finalize (object);
}

static void
plugin_jhbuild_sdk_class_init (PluginJhbuildSdkClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundrySdkClass *sdk_class = FOUNDRY_SDK_CLASS (klass);

  object_class->finalize = plugin_jhbuild_sdk_finalize;

  sdk_class->prepare_to_build = plugin_jhbuild_sdk_prepare_to_build;
  sdk_class->prepare_to_run = plugin_jhbuild_sdk_prepare_to_run;
  sdk_class->dup_config_option = plugin_jhbuild_sdk_dup_config_option;
}

static void
plugin_jhbuild_sdk_init (PluginJhbuildSdk *self)
{
  foundry_sdk_set_id (FOUNDRY_SDK (self), "jhbuild");
  foundry_sdk_set_name (FOUNDRY_SDK (self), "JHBuild");
  foundry_sdk_set_installed (FOUNDRY_SDK (self), TRUE);
  foundry_sdk_set_arch (FOUNDRY_SDK (self), foundry_get_default_arch ());
  foundry_sdk_set_kind (FOUNDRY_SDK (self), "jhbuild");
}

FoundrySdk *
plugin_jhbuild_sdk_new (FoundryContext *context,
                        const char     *install_prefix,
                        const char     *library_dir)
{
  PluginJhbuildSdk *self;

  self = g_object_new (PLUGIN_TYPE_JHBUILD_SDK,
                       "context", context,
                       NULL);
  self->install_prefix = g_strdup (install_prefix);
  self->library_dir = g_strdup (library_dir);

  return FOUNDRY_SDK (self);
}

char *
plugin_jhbuild_sdk_dup_install_prefix (PluginJhbuildSdk *self)
{
  g_return_val_if_fail (PLUGIN_IS_JHBUILD_SDK (self), NULL);

  return g_strdup (self->install_prefix);
}
