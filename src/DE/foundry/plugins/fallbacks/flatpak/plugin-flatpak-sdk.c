/* plugin-flatpak-sdk.c
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

#include <glib/gi18n-lib.h>

#include "plugin-flatpak.h"
#include "plugin-flatpak-aux.h"
#include "plugin-flatpak-config.h"
#include "plugin-flatpak-sdk-private.h"
#include "plugin-flatpak-util.h"

enum {
  PROP_0,
  PROP_INSTALLATION,
  PROP_REF,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (PluginFlatpakSdk, plugin_flatpak_sdk, FOUNDRY_TYPE_SDK)

static GParamSpec *properties[N_PROPS];

typedef struct _ContainsProgram
{
  PluginFlatpakSdk *sdk;
  char *program;
} ContainsProgram;

static void
contains_program_free (ContainsProgram *state)
{
  g_clear_object (&state->sdk);
  g_clear_pointer (&state->program, g_free);
  g_free (state);
}

static DexFuture *
plugin_flatpak_sdk_contains_program_fiber (gpointer data)
{
  static const char *known_path_dirs[] = { "/bin" };
  ContainsProgram *state = data;
  g_autofree char *deploy_dir = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (state != NULL);
  g_assert (PLUGIN_IS_FLATPAK_SDK (state->sdk));
  g_assert (state->program != NULL);
  g_assert (FLATPAK_IS_INSTALLED_REF (state->sdk->ref));

  deploy_dir = g_strdup (flatpak_installed_ref_get_deploy_dir (FLATPAK_INSTALLED_REF (state->sdk->ref)));

  for (guint i = 0; i < G_N_ELEMENTS (known_path_dirs); i++)
    {
      g_autofree char *outside_path = NULL;

      outside_path = g_build_filename (deploy_dir,
                                       "files",
                                       known_path_dirs[i],
                                       state->program,
                                       NULL);

      /* Check that the file exists instead of things like IS_EXECUTABLE.  The
       * reason we MUST check for either EXISTS or _IS_SYMLINK separately is
       * that EXISTS will check that the destination file exists too. That may
       * not be possible until the mount namespaces are correctly setup.
       *
       * See https://gitlab.gnome.org/GNOME/gnome-builder/-/issues/2050#note_1841120
       */
      if (dex_await_boolean (foundry_file_test (outside_path, G_FILE_TEST_IS_SYMLINK), NULL) ||
          dex_await_boolean (foundry_file_test (outside_path, G_FILE_TEST_EXISTS), NULL))
        {
          g_autofree char *inside_path = g_build_filename (known_path_dirs[i], state->program, NULL);
          foundry_path_cache_insert (state->sdk->path_cache, state->program, inside_path);
          return dex_future_new_take_string (g_steal_pointer (&inside_path));
        }
    }

  foundry_path_cache_insert (state->sdk->path_cache, state->program, NULL);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Program \"%s\" could not be found",
                                state->program);
}

static DexFuture *
plugin_flatpak_sdk_contains_program (FoundrySdk *sdk,
                                     const char *program)
{
  PluginFlatpakSdk *self = (PluginFlatpakSdk *)sdk;
  g_autofree char *path = NULL;
  ContainsProgram *state;

  g_assert (PLUGIN_IS_FLATPAK_SDK (self));
  g_assert (program != NULL);

  if (foundry_path_cache_lookup (self->path_cache, program, &path))
    return dex_future_new_take_string (g_steal_pointer (&path));

  state = g_new0 (ContainsProgram, 1);
  state->sdk = g_object_ref (self);
  state->program = g_strdup (program);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_sdk_contains_program_fiber,
                              state,
                              (GDestroyNotify) contains_program_free);
}

static char *
join_paths (const char *prepend,
            const char *path,
            const char *append)
{
  g_autofree char *tmp = foundry_search_path_prepend (path, prepend);
  return foundry_search_path_append (tmp, append);
}

typedef struct _Prepare
{
  PluginFlatpakSdk          *self;
  FoundryBuildPipeline      *pipeline;
  FoundryContext            *context;
  FoundryConfig             *config;
  FoundryBuildPipelinePhase  phase;
} Prepare;

static void
prepare_free (Prepare *prepare)
{
  g_clear_object (&prepare->self);
  g_clear_object (&prepare->context);
  g_clear_object (&prepare->pipeline);
  g_clear_object (&prepare->config);
  g_free (prepare);
}

static gboolean
plugin_flatpak_sdk_handle_build_context_cb (FoundryProcessLauncher  *launcher,
                                            const char * const      *argv,
                                            const char * const      *env,
                                            const char              *cwd,
                                            FoundryUnixFDMap        *unix_fd_map,
                                            gpointer                 user_data,
                                            GError                 **error)
{
  Prepare *prepare = user_data;
  g_autoptr(GFile) state_dir = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_autofree char *staging_dir = NULL;
  g_autofree char *cache_root = NULL;
  g_autofree char *ccache_dir = NULL;
  g_autofree char *prepend_path = NULL;
  g_autofree char *append_path = NULL;
  g_autofree char *new_path = NULL;
  const char *path;

  g_assert (prepare != NULL);
  g_assert (PLUGIN_IS_FLATPAK_SDK (prepare->self));
  g_assert (FOUNDRY_IS_CONTEXT (prepare->context));
  g_assert (FOUNDRY_IS_CONFIG (prepare->config));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (prepare->pipeline));

  /* Pass through the FD mappings */
  if (!foundry_process_launcher_merge_unix_fd_map (launcher, unix_fd_map, error))
    return FALSE;

  staging_dir = plugin_flatpak_get_staging_dir (prepare->pipeline);
  project_dir = foundry_context_dup_project_directory (prepare->context);
  state_dir = foundry_context_dup_state_directory (prepare->context);

  /* Pass CWD through */
  foundry_process_launcher_set_cwd (launcher, cwd);

  /* Setup FLATPAK_CONFIG_DIR= */
  plugin_flatpak_apply_config_dir (prepare->context, launcher);

  /* Setup CCACHE_DIR environment */
  cache_root = foundry_context_cache_filename (prepare->context, NULL);
  ccache_dir = foundry_context_cache_filename (prepare->context, "ccache", NULL);
  foundry_process_launcher_setenv (launcher, "CCACHE_DIR", ccache_dir);

  /* We want some environment available to the `flatpak build` environment
   * so that we can have working termcolor support.
   */
  foundry_process_launcher_setenv (launcher, "TERM", "xterm-256color");
  foundry_process_launcher_setenv (launcher, "COLORTERM", "truecolor");

  /* Now setup our basic arguments for the application */
  foundry_process_launcher_append_argv (launcher, "flatpak");
  foundry_process_launcher_append_argv (launcher, "build");
  foundry_process_launcher_append_argv (launcher, "--with-appdir");
  foundry_process_launcher_append_argv (launcher, "--allow=devel");
  foundry_process_launcher_append_argv (launcher, "--die-with-parent");

  /* Setup various directory access */
  foundry_process_launcher_append_formatted (launcher, "--filesystem=%s", g_file_peek_path (project_dir));
  foundry_process_launcher_append_formatted (launcher, "--filesystem=%s", g_file_peek_path (state_dir));
  foundry_process_launcher_append_formatted (launcher, "--filesystem=%s", cache_root);
  foundry_process_launcher_append_argv (launcher, "--nofilesystem=host");

  /* Restrict things when an actual flatpak manifest is used instead of
   * just selecting it as a SDK.
   */
  if (PLUGIN_IS_FLATPAK_CONFIG (prepare->config))
    {
      PluginFlatpakConfig *config = PLUGIN_FLATPAK_CONFIG (prepare->config);
      g_autoptr(FoundryFlatpakManifest) manifest = plugin_flatpak_config_dup_manifest (config);
      g_autoptr(FoundryFlatpakModule) primary_module = plugin_flatpak_config_dup_primary_module (config);
      g_autoptr(FoundryFlatpakOptions) options = foundry_flatpak_manifest_dup_build_options (manifest);

      /* If there are global build-args set, then we always apply them. */
      if (options != NULL)
        {
          g_auto(GStrv) build_args = foundry_flatpak_options_dup_build_args (options);

          if (build_args != NULL)
            foundry_process_launcher_append_args (launcher, (const char * const *)build_args);

          append_path = foundry_flatpak_options_dup_append_path (options);
          prepend_path = foundry_flatpak_options_dup_prepend_path (options);
        }

      /* If this is for a build system, then we also want to apply the build
       * args for the primary module.
       */
      if (prepare->phase == FOUNDRY_BUILD_PIPELINE_PHASE_BUILD && primary_module != NULL)
        {
          g_autoptr(FoundryFlatpakOptions) primary_build_options = foundry_flatpak_module_dup_build_options (primary_module);

          if (primary_build_options != NULL)
            {
              g_auto(GStrv) primary_build_args = foundry_flatpak_options_dup_build_args (primary_build_options);

              if (primary_build_args != NULL)
                foundry_process_launcher_append_args (launcher, (const char * const *)primary_build_args);
            }
        }
    }

  /* Always include `--share=network` because incremental building tends
   * to be different than one-shot building for a Flatpak build as developers
   * are likely to not have all the deps fetched via submodules they just
   * changed or even additional sources within the app's manifest module.
   *
   * See https://gitlab.gnome.org/GNOME/gnome-builder/-/issues/1775 for
   * more information. Having flatpak-builder as a library could allow us
   * to not require these sorts of workarounds.
   */
  if (!g_strv_contains (foundry_process_launcher_get_argv (launcher), "--share=network"))
    foundry_process_launcher_append_argv (launcher, "--share=network");

  /* Use an alternate PATH */
  if (!(path = g_environ_getenv ((char **)env, "PATH")))
    path = "/app/bin:/usr/bin";
  new_path = join_paths (prepend_path, path, append_path);

  /* Convert environment from upper level into --env=FOO=BAR */
  if (env != NULL)
    {
      for (guint i = 0; env[i]; i++)
        {
          if (new_path == NULL || !foundry_str_equal0 (env[i], "PATH"))
            foundry_process_launcher_append_formatted (launcher, "--env=%s", env[i]);
        }
    }

  if (new_path != NULL)
    foundry_process_launcher_append_formatted (launcher, "--env=PATH=%s", new_path);

  /* And last, before our child command, is the staging directory */
  foundry_process_launcher_append_argv (launcher, staging_dir);

  /* And now the upper layer's command arguments */
  foundry_process_launcher_append_args (launcher, argv);

  return TRUE;
}

static DexFuture *
plugin_flatpak_sdk_prepare_to_build (FoundrySdk                *sdk,
                                     FoundryBuildPipeline      *pipeline,
                                     FoundryProcessLauncher    *launcher,
                                     FoundryBuildPipelinePhase  phase)
{
  PluginFlatpakSdk *self = (PluginFlatpakSdk *)sdk;
  Prepare *prepare;

  g_assert (PLUGIN_IS_FLATPAK_SDK (self));
  g_assert (!pipeline || FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  /* Required for staging_dir */
  if (pipeline == NULL)
    return foundry_future_new_not_supported ();

  prepare = g_new0 (Prepare, 1);
  prepare->self = g_object_ref (self);
  prepare->pipeline = g_object_ref (pipeline);
  prepare->phase = phase;
  prepare->context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  prepare->config = foundry_build_pipeline_dup_config (pipeline);

  /* We have to run "flatpak build" from the host */
  foundry_process_launcher_push_host (launcher);

  /* Handle the upper layer to rewrite the command using "flatpak build" */
  foundry_process_launcher_push (launcher,
                                 plugin_flatpak_sdk_handle_build_context_cb,
                                 prepare,
                                 (GDestroyNotify) prepare_free);

  return dex_future_new_true ();
}

static gboolean
can_pass_through_finish_arg (const char *arg)
{
  if (arg == NULL)
    return FALSE;

  return g_str_has_prefix (arg, "--allow") ||
         g_str_has_prefix (arg, "--share") ||
         g_str_has_prefix (arg, "--socket") ||
         g_str_has_prefix (arg, "--filesystem") ||
         g_str_has_prefix (arg, "--device") ||
         g_str_has_prefix (arg, "--env") ||
         g_str_has_prefix (arg, "--system-talk") ||
         g_str_has_prefix (arg, "--own-name") ||
         g_str_has_prefix (arg, "--talk-name") ||
         g_str_has_prefix (arg, "--add-policy");
}

static gboolean
maybe_profiling (const char * const *argv)
{
  if (argv == NULL)
    return FALSE;

  for (guint i = 0; argv[i]; i++)
    {
      if (strstr (argv[i], "sysprof-agent"))
        return TRUE;
    }

   return FALSE;
}

static gboolean
plugin_flatpak_sdk_handle_run_context_cb (FoundryProcessLauncher  *launcher,
                                          const char * const      *argv,
                                          const char * const      *env,
                                          const char              *cwd,
                                          FoundryUnixFDMap        *unix_fd_map,
                                          gpointer                 user_data,
                                          GError                 **error)
{
  Prepare *prepare = user_data;
  g_autoptr(GFile) state_dir = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_autofree char *staging_dir = NULL;
  g_autofree char *new_path = NULL;
  g_autofree char *app_id = NULL;

  g_assert (prepare != NULL);
  g_assert (PLUGIN_IS_FLATPAK_SDK (prepare->self));
  g_assert (FOUNDRY_IS_CONTEXT (prepare->context));
  g_assert (FOUNDRY_IS_CONFIG (prepare->config));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (prepare->pipeline));

  /* Pass through the FD mappings */
  if (!foundry_process_launcher_merge_unix_fd_map (launcher, unix_fd_map, error))
    return FALSE;

  staging_dir = plugin_flatpak_get_staging_dir (prepare->pipeline);
  project_dir = foundry_context_dup_project_directory (prepare->context);
  state_dir = foundry_context_dup_state_directory (prepare->context);

  if (PLUGIN_IS_FLATPAK_CONFIG (prepare->config))
    app_id = plugin_flatpak_config_dup_id (PLUGIN_FLATPAK_CONFIG (prepare->config));

  /* Pass CWD through */
  foundry_process_launcher_set_cwd (launcher, cwd);

  /* Setup a minimal environment for running (DISPLAY, etc) */
  foundry_process_launcher_add_minimal_environment (launcher);

  /* Setup FLATPAK_CONFIG_DIR= */
  plugin_flatpak_apply_config_dir (prepare->context, launcher);

  /* Now setup our basic arguments for the application */
  foundry_process_launcher_append_argv (launcher, "flatpak");
  foundry_process_launcher_append_argv (launcher, "build");
  foundry_process_launcher_append_argv (launcher, "--with-appdir");
  foundry_process_launcher_append_argv (launcher, "--allow=devel");
  foundry_process_launcher_append_argv (launcher, "--die-with-parent");

  /* Make sure we have access to the document portal */
  if (app_id != NULL)
    foundry_process_launcher_append_formatted (launcher,
                                               "--bind-mount=/run/user/%u/doc=/run/user/%u/doc/by-app/%s",
                                               getuid (), getuid (), app_id);

  /* Make sure we have access to fonts and such */
  plugin_flatpak_aux_append_to_launcher (launcher);

  /* Setup various directory access in case what is being run requires them */
  foundry_process_launcher_append_formatted (launcher, "--filesystem=%s", g_file_peek_path (project_dir));
  foundry_process_launcher_append_formatted (launcher, "--filesystem=%s", g_file_peek_path (state_dir));
  foundry_process_launcher_append_argv (launcher, "--nofilesystem=host");

  /* Convert environment from upper level into --env=FOO=BAR */
  if (env != NULL)
    {
      for (guint i = 0; env[i]; i++)
        foundry_process_launcher_append_formatted (launcher, "--env=%s", env[i]);
    }


  if (PLUGIN_IS_FLATPAK_CONFIG (prepare->config))
    {
      PluginFlatpakConfig *config = PLUGIN_FLATPAK_CONFIG (prepare->config);
      g_autoptr(FoundryFlatpakManifest) manifest = plugin_flatpak_config_dup_manifest (config);
      g_auto(GStrv) finish_args = foundry_flatpak_manifest_dup_finish_args (manifest);

      if (finish_args != NULL)
        {
          for (guint i = 0; finish_args[i]; i++)
            {
              if (can_pass_through_finish_arg (finish_args[i]))
                foundry_process_launcher_append_argv (launcher, finish_args[i]);
            }
        }
    }
  else
    {
      foundry_process_launcher_append_argv (launcher, "--share=ipc");
      foundry_process_launcher_append_argv (launcher, "--share=network");
      foundry_process_launcher_append_argv (launcher, "--socket=x11");
      foundry_process_launcher_append_argv (launcher, "--socket=wayland");
    }

  /* Give access to portals */
  foundry_process_launcher_append_argv (launcher, "--talk-name=org.freedesktop.portal.*");

  /* Layering violation, but always give access to profiler */
  if (maybe_profiling (argv))
    {
      foundry_process_launcher_append_argv (launcher, "--system-talk-name=org.gnome.Sysprof3");
      foundry_process_launcher_append_argv (launcher, "--system-talk-name=org.freedesktop.PolicyKit1");
      foundry_process_launcher_append_argv (launcher, "--filesystem=~/.local/share/flatpak:ro");
      foundry_process_launcher_append_argv (launcher, "--filesystem=host");
    }

  /* Make A11y bus available to the application */
  foundry_process_launcher_append_argv (launcher, "--talk-name=org.a11y.Bus");

  {
    g_autofree char *a11y_bus = NULL;
    g_autofree char *a11y_bus_unix_path = NULL;
    g_autofree char *a11y_bus_address_suffix = NULL;

    if ((a11y_bus = dex_await_string (plugin_flatpak_get_a11y_bus (), NULL)) &&
        plugin_flatpak_parse_a11y_bus (a11y_bus, &a11y_bus_unix_path, &a11y_bus_address_suffix))
      {
        foundry_process_launcher_append_formatted (launcher,
                                                   "--bind-mount=/run/flatpak/at-spi-bus=%s",
                                                   a11y_bus_unix_path);
        foundry_process_launcher_append_formatted (launcher,
                                                   "--env=AT_SPI_BUS_ADDRESS=unix:path=/run/flatpak/at-spi-bus%s",
                                                   a11y_bus_address_suffix ? a11y_bus_address_suffix : "");
      }
  }

  /* Make sure we have access to user installed fonts for plugin-flatpak-aux.c */
  foundry_process_launcher_append_argv (launcher, "--filesystem=~/.local/share/fonts:ro");

  /* And last, before our child command, is the staging directory */
  foundry_process_launcher_append_argv (launcher, staging_dir);

  /* And now the upper layer's command arguments */
  foundry_process_launcher_append_args (launcher, argv);

  return TRUE;
}

static DexFuture *
plugin_flatpak_sdk_prepare_to_run (FoundrySdk             *sdk,
                                   FoundryBuildPipeline   *pipeline,
                                   FoundryProcessLauncher *launcher)
{
  PluginFlatpakSdk *self = (PluginFlatpakSdk *)sdk;
  Prepare *prepare;

  g_assert (PLUGIN_IS_FLATPAK_SDK (self));
  g_assert (FOUNDRY_IS_BUILD_PIPELINE (pipeline));
  g_assert (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  prepare = g_new0 (Prepare, 1);
  prepare->self = g_object_ref (self);
  prepare->pipeline = g_object_ref (pipeline);
  prepare->phase = 0;
  prepare->context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  prepare->config = foundry_build_pipeline_dup_config (pipeline);

  /* We have to run "flatpak build" from the host */
  foundry_process_launcher_push_host (launcher);

  /* Handle the upper layer to rewrite the command using "flatpak build" */
  foundry_process_launcher_push (launcher,
                                 plugin_flatpak_sdk_handle_run_context_cb,
                                 prepare,
                                 (GDestroyNotify) prepare_free);

  return dex_future_new_true ();
}

static void
plugin_flatpak_sdk_constructed (GObject *object)
{
  PluginFlatpakSdk *self = (PluginFlatpakSdk *)object;
  g_autofree char *id = NULL;
  g_autofree char *sdk_name = NULL;
  const char *name;
  const char *arch;
  const char *branch;

  G_OBJECT_CLASS (plugin_flatpak_sdk_parent_class)->constructed (object);

  if (self->installation == NULL || self->ref == NULL)
    g_return_if_reached ();

  name = flatpak_ref_get_name (self->ref);
  arch = flatpak_ref_get_arch (self->ref);
  branch = flatpak_ref_get_branch (self->ref);

  id = g_strdup_printf ("%s/%s/%s", name, arch, branch);

  if (g_str_equal (flatpak_get_default_arch (), arch))
    sdk_name = g_strdup_printf ("%s %s", name, branch);
  else
    sdk_name = g_strdup_printf ("%s %s (%s)", name, branch, arch);

  foundry_sdk_set_id (FOUNDRY_SDK (self), id);
  foundry_sdk_set_name (FOUNDRY_SDK (self), sdk_name);
  foundry_sdk_set_kind (FOUNDRY_SDK (self), "flatpak");
  foundry_sdk_set_arch (FOUNDRY_SDK (self), arch);

  if (FLATPAK_IS_INSTALLED_REF (self->ref))
    foundry_sdk_set_installed (FOUNDRY_SDK (self), TRUE);
}

static char *
plugin_flatpak_sdk_dup_config_option (FoundrySdk             *sdk,
                                      FoundrySdkConfigOption  option)
{
  g_assert (PLUGIN_IS_FLATPAK_SDK (sdk));

  switch (option)
    {
    case FOUNDRY_SDK_CONFIG_OPTION_PREFIX:
      return g_strdup ("/app");

    case FOUNDRY_SDK_CONFIG_OPTION_LIBDIR:
      return g_strdup ("lib");

    default:
      return NULL;
    }
}

static void
plugin_flatpak_sdk_finalize (GObject *object)
{
  PluginFlatpakSdk *self = (PluginFlatpakSdk *)object;

  g_clear_object (&self->path_cache);
  g_clear_object (&self->installation);
  g_clear_object (&self->ref);

  G_OBJECT_CLASS (plugin_flatpak_sdk_parent_class)->finalize (object);
}

static void
plugin_flatpak_sdk_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  PluginFlatpakSdk *self = PLUGIN_FLATPAK_SDK (object);

  switch (prop_id)
    {
    case PROP_INSTALLATION:
      g_value_set_object (value, self->installation);
      break;

    case PROP_REF:
      g_value_set_object (value, self->ref);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_sdk_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  PluginFlatpakSdk *self = PLUGIN_FLATPAK_SDK (object);

  switch (prop_id)
    {
    case PROP_INSTALLATION:
      self->installation = g_value_dup_object (value);
      break;

    case PROP_REF:
      self->ref = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_flatpak_sdk_class_init (PluginFlatpakSdkClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundrySdkClass *sdk_class = FOUNDRY_SDK_CLASS (klass);

  object_class->constructed = plugin_flatpak_sdk_constructed;
  object_class->finalize = plugin_flatpak_sdk_finalize;
  object_class->get_property = plugin_flatpak_sdk_get_property;
  object_class->set_property = plugin_flatpak_sdk_set_property;

  sdk_class->install = plugin_flatpak_sdk_install;
  sdk_class->contains_program = plugin_flatpak_sdk_contains_program;
  sdk_class->prepare_to_build = plugin_flatpak_sdk_prepare_to_build;
  sdk_class->prepare_to_run = plugin_flatpak_sdk_prepare_to_run;
  sdk_class->dup_config_option = plugin_flatpak_sdk_dup_config_option;

  properties[PROP_INSTALLATION] =
    g_param_spec_object ("installation", NULL, NULL,
                         FLATPAK_TYPE_INSTALLATION,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_REF] =
    g_param_spec_object ("ref", NULL, NULL,
                         FLATPAK_TYPE_REF,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  plugin_flatpak_aux_init ();
}

static void
plugin_flatpak_sdk_init (PluginFlatpakSdk *self)
{
  self->path_cache = foundry_path_cache_new ();
}

PluginFlatpakSdk *
plugin_flatpak_sdk_new (FoundryContext      *context,
                        FlatpakInstallation *installation,
                        FlatpakRef          *ref)
{
  gboolean extension_only;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (FLATPAK_IS_INSTALLATION (installation), NULL);
  g_return_val_if_fail (FLATPAK_IS_REF (ref), NULL);

  /* Really we need to check this by looking at the metadata bytes. But
   * this is much faster than doing that and generally gets the same
   * answer.
   */
  extension_only = strstr (flatpak_ref_get_name (ref), ".Extension.") != NULL;

  return g_object_new (PLUGIN_TYPE_FLATPAK_SDK,
                       "context", context,
                       "extension-only", extension_only,
                       "installation", installation,
                       "ref", ref,
                       NULL);
}
