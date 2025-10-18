/* plugin-shellcheck-diagnostic-tool.c
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

#include <glib/gi18n-lib.h>

#include "plugin-shellcheck-diagnostic-tool.h"

struct _PluginShellcheckDiagnosticTool
{
  FoundryDiagnosticTool parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginShellcheckDiagnosticTool, plugin_shellcheck_diagnostic_tool, FOUNDRY_TYPE_DIAGNOSTIC_TOOL)

static inline FoundryDiagnosticSeverity
parse_severity (const char *level)
{
  if (foundry_str_equal0 (level, "error"))
    return FOUNDRY_DIAGNOSTIC_ERROR;

  if (foundry_str_equal0 (level, "warning"))
    return FOUNDRY_DIAGNOSTIC_WARNING;

  if (foundry_str_equal0 (level, "info"))
    return FOUNDRY_DIAGNOSTIC_NOTE;

  if (foundry_str_equal0 (level, "style"))
    return FOUNDRY_DIAGNOSTIC_NOTE;

  return FOUNDRY_DIAGNOSTIC_NOTE;
}

static DexFuture *
plugin_shellcheck_diagnostic_tool_dup_bytes_for_stdin (FoundryDiagnosticTool *diagnostic_tool,
                                                       GFile                 *file,
                                                       GBytes                *contents,
                                                       const char            *laguage)
{
  g_assert (PLUGIN_IS_SHELLCHECK_DIAGNOSTIC_TOOL (diagnostic_tool));
  g_assert (!file || G_IS_FILE (file));
  g_assert (file || contents);

  if (contents == NULL)
    return dex_file_load_contents_bytes (file);
  else
    return dex_future_new_take_boxed (G_TYPE_BYTES, g_bytes_ref (contents));
}

static DexFuture *
plugin_shellcheck_diagnostic_tool_extract_from_stdout (FoundryDiagnosticTool *diagnostic_tool,
                                                       GFile                 *file,
                                                       GBytes                *contents,
                                                       const char            *language,
                                                       GBytes                *stdout_bytes)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(JsonParser) parser = NULL;
  g_autoptr(GError) error = NULL;
  JsonArray *results;
  JsonNode *root;

  g_assert (PLUGIN_IS_SHELLCHECK_DIAGNOSTIC_TOOL (diagnostic_tool));
  g_assert (!file || G_IS_FILE (file));
  g_assert (file || contents);
  g_assert (stdout_bytes);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (diagnostic_tool));
  store = g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC);

#if 0
  [{"file":"-","line":1,"endLine":1,"column":1,"endColumn":1,"level":"error","code":1073,"message":"Couldn't parse this function. Fix to allow more checks.","fix":null},
    {"file":"-","line":1,"endLine":1,"column":7,"endColumn":7,"level":"error","code":1064,"message":"Expected a { to open the function definition.","fix":null},
    {"file":"-","line":1,"endLine":1,"column":7,"endColumn":7,"level":"error","code":1072,"message":"Fix any mentioned problems and try again.","fix":null}]
#endif

  parser = json_parser_new ();

  if (!json_parser_load_from_data (parser,
                                   g_bytes_get_data (stdout_bytes, NULL),
                                   g_bytes_get_size (stdout_bytes),
                                   &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if ((root = json_parser_get_root (parser)) &&
      JSON_NODE_HOLDS_ARRAY (root) &&
      (results = json_node_get_array (root)))
    {
      guint n_results = json_array_get_length (results);

      for (guint r = 0; r < n_results; r++)
        {
          JsonObject *message = json_array_get_object_element (results, r);
          g_autoptr(FoundryDiagnosticBuilder) builder = NULL;
          g_autoptr(FoundryDiagnostic) diagnostic = NULL;
          FoundryDiagnosticSeverity severity;
          const char *level;
          guint start_line;
          guint start_col;

          if (!json_object_has_member (message, "file") ||
              !json_object_has_member (message, "line"))
            continue;

          builder = foundry_diagnostic_builder_new (context);
          foundry_diagnostic_builder_set_file (builder, file);

          start_line = MAX (json_object_get_int_member (message, "line"), 1);
          start_col = MAX (json_object_get_int_member (message, "column"), 1);

          foundry_diagnostic_builder_set_line (builder, start_line);
          foundry_diagnostic_builder_set_line_offset (builder, start_col);

          if (json_object_has_member (message, "endLine") &&
              json_object_has_member (message, "endColumn"))
            {
              guint end_line = MAX (json_object_get_int_member (message, "endLine"), 1);
              guint end_col = MAX (json_object_get_int_member (message, "endColumn"), 1);

              foundry_diagnostic_builder_add_range (builder, start_line, start_col, end_line, end_col);
            }

          if ((level = json_object_get_string_member (message, "level")))
            severity = parse_severity (level);
          else
            severity = FOUNDRY_DIAGNOSTIC_ERROR;

          foundry_diagnostic_builder_set_severity (builder, severity);
          foundry_diagnostic_builder_set_message (builder,
                                                  json_object_get_string_member (message, "message"));

          diagnostic = foundry_diagnostic_builder_end (builder);
          g_list_store_append (store, diagnostic);
        }
    }


  return dex_future_new_take_object (g_steal_pointer (&store));
}

static void
plugin_shellcheck_diagnostic_tool_class_init (PluginShellcheckDiagnosticToolClass *klass)
{
  FoundryDiagnosticToolClass *diagnostic_tool_class = FOUNDRY_DIAGNOSTIC_TOOL_CLASS (klass);

  diagnostic_tool_class->dup_bytes_for_stdin = plugin_shellcheck_diagnostic_tool_dup_bytes_for_stdin;
  diagnostic_tool_class->extract_from_stdout = plugin_shellcheck_diagnostic_tool_extract_from_stdout;
}

static void
plugin_shellcheck_diagnostic_tool_init (PluginShellcheckDiagnosticTool *self)
{
  foundry_diagnostic_tool_set_argv (FOUNDRY_DIAGNOSTIC_TOOL (self),
                                    FOUNDRY_STRV_INIT ("shellcheck", "--format=json", "-"));
}
