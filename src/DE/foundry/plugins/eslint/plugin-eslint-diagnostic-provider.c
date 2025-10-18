/* plugin-eslint-diagnostic-provider.c
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

#include "plugin-eslint-diagnostic-provider.h"

struct _PluginEslintDiagnosticProvider
{
  FoundryDiagnosticProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginEslintDiagnosticProvider, plugin_eslint_diagnostic_provider, FOUNDRY_TYPE_DIAGNOSTIC_PROVIDER)

static inline FoundryDiagnosticSeverity
parse_severity (int n)
{
  switch (n)
    {
    case 1:
      return FOUNDRY_DIAGNOSTIC_WARNING;
    case 2:
      return FOUNDRY_DIAGNOSTIC_ERROR;
    default:
      return FOUNDRY_DIAGNOSTIC_NOTE;
    }
}

static void
parse_results (FoundryContext *context,
               GListStore     *store,
               GBytes         *stdout_bytes,
               GFile          *file)
{
  g_autoptr(JsonParser) parser = NULL;
  g_autoptr(GError) error = NULL;
  JsonNode *root;
  JsonArray *results;

  g_assert (FOUNDRY_IS_CONTEXT (context));
  g_assert (G_IS_LIST_STORE (store));
  g_assert (stdout_bytes != NULL);
  g_assert (!file || G_IS_FILE (file));

  parser = json_parser_new ();

  if (!json_parser_load_from_data (parser,
                                   g_bytes_get_data (stdout_bytes, NULL),
                                   g_bytes_get_size (stdout_bytes),
                                   &error))
    {
      g_debug ("Failed to parse eslint result: %s", error->message);
      return;
    }

  if ((root = json_parser_get_root (parser)) &&
      JSON_NODE_HOLDS_ARRAY (root) &&
      (results = json_node_get_array (root)))
    {
      guint n_results = json_array_get_length (results);

      for (guint r = 0; r < n_results; r++)
        {
          g_autoptr(FoundryDiagnosticBuilder) builder = foundry_diagnostic_builder_new (context);
          JsonObject *result = json_array_get_object_element (results, r);
          JsonArray *messages = json_object_get_array_member (result, "messages");
          guint n_messages = json_array_get_length (messages);

          for (guint m = 0; m < n_messages; m++)
            {
              JsonObject *message = json_array_get_object_element (messages, m);
              g_autoptr(FoundryDiagnostic) diagnostic = NULL;
              FoundryDiagnosticSeverity severity;
              guint start_line;
              guint start_col;

              if (!json_object_has_member (message, "line") ||
                  !json_object_has_member (message, "column"))
                continue;

              start_line = MAX (json_object_get_int_member (message, "line"), 1);
              start_col = MAX (json_object_get_int_member (message, "column"), 1);

              foundry_diagnostic_builder_set_line (builder, start_line);
              foundry_diagnostic_builder_set_line_offset (builder, start_col);

              if (file != NULL)
                foundry_diagnostic_builder_set_file (builder, file);

              if (json_object_has_member (message, "endLine") &&
                  json_object_has_member (message, "endColumn"))
                {
                  guint end_line = MAX (json_object_get_int_member (message, "endLine"), 1);
                  guint end_col = MAX (json_object_get_int_member (message, "endColumn"), 1);

                  foundry_diagnostic_builder_add_range (builder, start_line, start_col, end_line, end_col);
                }

              severity = parse_severity (json_object_get_int_member (message, "severity"));
              foundry_diagnostic_builder_set_severity (builder, severity);

              foundry_diagnostic_builder_set_message (builder,
                                                      json_object_get_string_member (message, "message"));

              /* TODO: (from python implementation)
               *
               * if 'fix' in message:
               * Fixes often come without end* information so we
               * will rarely get here, instead it has a file offset
               * which is not actually implemented in IdeSourceLocation
               * fixit = Ide.Fixit.new(range_, message['fix']['text'])
               * diagnostic.take_fixit(fixit)
               */

              if ((diagnostic = foundry_diagnostic_builder_end (builder)))
                g_list_store_append (store, diagnostic);
            }
        }
    }
}

static DexFuture *
plugin_eslint_diagnostic_provider_diagnose_fiber (FoundryContext *context,
                                                  GFile          *file,
                                                  GBytes         *contents,
                                                  const char     *language)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GBytes) stdout_bytes = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_autoptr(GFile) bin_eslint = NULL;
  GSubprocessFlags flags = 0;
  const char *command = "eslint";

  g_assert (FOUNDRY_IS_CONTEXT (context));
  g_assert (!file || G_IS_FILE (file));

  build_manager = foundry_context_dup_build_manager (context);
  pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), NULL);

  project_dir = foundry_context_dup_project_directory (context);
  bin_eslint = g_file_get_child (project_dir, "node_modules/.bin/eslint");

  if (dex_await_boolean (dex_file_query_exists (bin_eslint), NULL))
    command = g_file_peek_path (bin_eslint);

  launcher = foundry_process_launcher_new ();

  if (pipeline &&
      !dex_await (foundry_build_pipeline_prepare (pipeline, launcher, FOUNDRY_BUILD_PIPELINE_PHASE_BUILD), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  foundry_process_launcher_append_argv (launcher, command);
  foundry_process_launcher_append_args (launcher, FOUNDRY_STRV_INIT ("-f", "json"));
  foundry_process_launcher_append_args (launcher, FOUNDRY_STRV_INIT ("--ignore-pattern", "!node_modules/*"));
  foundry_process_launcher_append_args (launcher, FOUNDRY_STRV_INIT ("--ignore-pattern", "!bower_components/*"));

  if (contents != NULL)
    foundry_process_launcher_append_argv (launcher, "--stdin");

  if (file != NULL)
    {
      foundry_process_launcher_append_argv (launcher, "--stdin-filename");
      foundry_process_launcher_append_argv (launcher, g_file_peek_path (file));
    }

  flags = G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDERR_SILENCE;

  if (contents != NULL)
    flags |= G_SUBPROCESS_FLAGS_STDIN_PIPE;

  if (!(subprocess = foundry_process_launcher_spawn_with_flags (launcher, flags, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(stdout_bytes = dex_await_boxed (foundry_subprocess_communicate (subprocess, contents), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  store = g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC);

  parse_results (context, store, stdout_bytes, file);

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
plugin_eslint_diagnostic_provider_diagnose (FoundryDiagnosticProvider *provider,
                                            GFile                     *file,
                                            GBytes                    *contents,
                                            const char                *language)
{
  g_autoptr(FoundryContext) context = NULL;

  g_assert (PLUGIN_IS_ESLINT_DIAGNOSTIC_PROVIDER (provider));
  g_assert (!file || G_IS_FILE (file));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_eslint_diagnostic_provider_diagnose_fiber),
                                  4,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  G_TYPE_FILE, file,
                                  G_TYPE_BYTES, contents,
                                  G_TYPE_STRING, language);
}

static void
plugin_eslint_diagnostic_provider_class_init (PluginEslintDiagnosticProviderClass *klass)
{
  FoundryDiagnosticProviderClass *diagnostic_provider_class = FOUNDRY_DIAGNOSTIC_PROVIDER_CLASS (klass);

  diagnostic_provider_class->diagnose = plugin_eslint_diagnostic_provider_diagnose;
}

static void
plugin_eslint_diagnostic_provider_init (PluginEslintDiagnosticProvider *self)
{
}
