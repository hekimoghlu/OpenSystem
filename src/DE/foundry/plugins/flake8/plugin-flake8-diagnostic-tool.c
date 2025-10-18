/* plugin-flake8-diagnostic-tool.c
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

#include "line-reader-private.h"

#include "plugin-flake8-diagnostic-tool.h"

#define FLAKE8_DEFAULT_FORMAT "^(?<filename>[^:]+):(?<line>\\d+):(?<column>\\d+):\\s+(?<code>[^\\s]+)\\s+(?<text>.*)$"

struct _PluginFlake8DiagnosticTool
{
  FoundryDiagnosticTool parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginFlake8DiagnosticTool, plugin_flake8_diagnostic_tool, FOUNDRY_TYPE_DIAGNOSTIC_TOOL)

static inline FoundryDiagnosticSeverity
parse_severity (const char *code)
{
  switch (code[0])
    {
    case 'F':
      return FOUNDRY_DIAGNOSTIC_FATAL;
    case 'E':
      return FOUNDRY_DIAGNOSTIC_ERROR;
    case 'W':
      return FOUNDRY_DIAGNOSTIC_WARNING;
    case 'I':
      return FOUNDRY_DIAGNOSTIC_NOTE;
    default:
      return FOUNDRY_DIAGNOSTIC_NOTE;
    }
}

static DexFuture *
plugin_flake8_diagnostic_tool_dup_bytes_for_stdin (FoundryDiagnosticTool *diagnostic_tool,
                                                   GFile                 *file,
                                                   GBytes                *contents,
                                                   const char            *laguage)
{
  g_assert (PLUGIN_IS_FLAKE8_DIAGNOSTIC_TOOL (diagnostic_tool));
  g_assert (!file || G_IS_FILE (file));
  g_assert (file || contents);

  if (contents == NULL)
    return dex_file_load_contents_bytes (file);
  else
    return dex_future_new_take_boxed (G_TYPE_BYTES, g_bytes_ref (contents));
}

static DexFuture *
plugin_flake8_diagnostic_tool_extract_from_stdout (FoundryDiagnosticTool *diagnostic_tool,
                                                   GFile                 *file,
                                                   GBytes                *contents,
                                                   const char            *language,
                                                   GBytes                *stdout_bytes)
{
  static GRegex *regex;

  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GError) error = NULL;
  LineReader reader;
  const char *l;
  gsize len;

  g_assert (PLUGIN_IS_FLAKE8_DIAGNOSTIC_TOOL (diagnostic_tool));
  g_assert (!file || G_IS_FILE (file));
  g_assert (file || contents);
  g_assert (stdout_bytes);

  if (regex == NULL)
    regex = g_regex_new (FLAKE8_DEFAULT_FORMAT, G_REGEX_OPTIMIZE, 0, NULL);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (diagnostic_tool));
  store = g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC);

  line_reader_init (&reader,
                    (char *)g_bytes_get_data (stdout_bytes, NULL),
                    g_bytes_get_size (stdout_bytes));
  while ((l = line_reader_next (&reader, &len)))
    {
      g_autoptr(FoundryDiagnosticBuilder) builder = NULL;
      g_autoptr(FoundryDiagnostic) diagnostic = NULL;
      g_autoptr(GMatchInfo) match_info = NULL;
      g_autofree char *filename = NULL;
      g_autofree char *line = NULL;
      g_autofree char *column = NULL;
      g_autofree char *code = NULL;
      g_autofree char *text = NULL;
      g_autofree char *message = NULL;
      FoundryDiagnosticSeverity severity;
      guint64 lineno;
      guint64 columnno;

      if (!g_regex_match_full (regex, l, len, 0, 0, &match_info, NULL) ||
          !g_match_info_matches (match_info))
        continue;

      filename = g_match_info_fetch (match_info, 1);
      line = g_match_info_fetch (match_info, 2);
      column = g_match_info_fetch (match_info, 3);
      code = g_match_info_fetch (match_info, 4);
      text = g_match_info_fetch (match_info, 5);

      severity = parse_severity (code);
      lineno = g_ascii_strtoull (line, NULL, 10);
      columnno = g_ascii_strtoull (column, NULL, 10);
      message = g_strdup_printf ("%s: %s", code, text);

      builder = foundry_diagnostic_builder_new (context);
      foundry_diagnostic_builder_set_file (builder, file);
      foundry_diagnostic_builder_set_line (builder, lineno);
      foundry_diagnostic_builder_set_line_offset (builder, columnno);
      foundry_diagnostic_builder_set_severity (builder, severity);
      foundry_diagnostic_builder_take_message (builder, g_steal_pointer (&message));

      diagnostic = foundry_diagnostic_builder_end (builder);

      g_list_store_append (store, diagnostic);
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static void
plugin_flake8_diagnostic_tool_class_init (PluginFlake8DiagnosticToolClass *klass)
{
  FoundryDiagnosticToolClass *diagnostic_tool_class = FOUNDRY_DIAGNOSTIC_TOOL_CLASS (klass);

  diagnostic_tool_class->dup_bytes_for_stdin = plugin_flake8_diagnostic_tool_dup_bytes_for_stdin;
  diagnostic_tool_class->extract_from_stdout = plugin_flake8_diagnostic_tool_extract_from_stdout;
}

static void
plugin_flake8_diagnostic_tool_init (PluginFlake8DiagnosticTool *self)
{
  foundry_diagnostic_tool_set_argv (FOUNDRY_DIAGNOSTIC_TOOL (self),
                                    FOUNDRY_STRV_INIT ("flake8", "--format=default", "-"));
}
