/* plugin-gettext-diagnostic-tool.c
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

#include <errno.h>

#include <glib/gi18n-lib.h>

#include "line-reader-private.h"

#include "plugin-gettext-diagnostic-tool.h"

struct _PluginGettextDiagnosticTool
{
  FoundryDiagnosticTool parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginGettextDiagnosticTool, plugin_gettext_diagnostic_tool, FOUNDRY_TYPE_DIAGNOSTIC_TOOL)

static DexFuture *
plugin_gettext_diagnostic_tool_dup_bytes_for_stdin (FoundryDiagnosticTool *diagnostic_tool,
                                                    GFile                 *file,
                                                    GBytes                *contents,
                                                    const char            *laguage)
{
  g_assert (PLUGIN_IS_GETTEXT_DIAGNOSTIC_TOOL (diagnostic_tool));
  g_assert (!file || G_IS_FILE (file));
  g_assert (file || contents);

  if (contents == NULL)
    return dex_file_load_contents_bytes (file);
  else
    return dex_future_new_take_boxed (G_TYPE_BYTES, g_bytes_ref (contents));
}

static DexFuture *
plugin_gettext_diagnostic_tool_extract_from_stdout (FoundryDiagnosticTool *diagnostic_tool,
                                                    GFile                 *file,
                                                    GBytes                *contents,
                                                    const char            *language,
                                                    GBytes                *stdout_bytes)
{
  g_autoptr(FoundryDiagnosticBuilder) builder = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GError) error = NULL;
  LineReader reader;
  const char *l;
  gsize len;

  g_assert (PLUGIN_IS_GETTEXT_DIAGNOSTIC_TOOL (diagnostic_tool));
  g_assert (!file || G_IS_FILE (file));
  g_assert (file || contents);
  g_assert (stdout_bytes);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (diagnostic_tool));
  store = g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC);

  builder = foundry_diagnostic_builder_new (context);
  foundry_diagnostic_builder_set_file (builder, file);

  line_reader_init (&reader,
                    (char *)g_bytes_get_data (stdout_bytes, NULL),
                    g_bytes_get_size (stdout_bytes));
  while ((l = line_reader_next (&reader, &len)))
    {
      g_autoptr(FoundryDiagnostic) diagnostic = NULL;
      g_autofree char *line = g_strndup (l, len);
      const char *ptr = line;
      guint64 lineno;

      /* Lines that we want to parse should look something like this:
       * "standard input:195: ASCII double quote used instead of Unicode"
       */

      if (!g_str_has_prefix (line, "standard input:"))
        continue;

      ptr += strlen ("standard input:");
      if (!g_ascii_isdigit (*ptr))
        continue;

      lineno = g_ascii_strtoull (ptr, (char **)&ptr, 10);
      if ((lineno == G_MAXUINT64 && errno == ERANGE) || ((lineno == 0) && errno == EINVAL))
        continue;

      if (!g_str_has_prefix (ptr, ": "))
        continue;

      ptr += strlen (": ");

      foundry_diagnostic_builder_set_line (builder, lineno);
      foundry_diagnostic_builder_set_message (builder, ptr);
      foundry_diagnostic_builder_set_severity (builder, FOUNDRY_DIAGNOSTIC_WARNING);

      diagnostic = foundry_diagnostic_builder_end (builder);

      g_list_store_append (store, diagnostic);
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
plugin_gettext_diagnostic_tool_prepare (FoundryDiagnosticTool  *tool,
                                        FoundryProcessLauncher *launcher,
                                        const char * const     *argv,
                                        const char * const     *environ,
                                        const char             *language)
{
  static const struct {
    const gchar *id;
    const gchar *lang;
  } id_to_lang[] = {
    { "awk", "awk" },
    { "c", "C" },
    { "chdr", "C" },
    { "cpp", "C++" },
    { "js", "JavaScript" },
    { "lisp", "Lisp" },
    { "objc", "ObjectiveC" },
    { "perl", "Perl" },
    { "php", "PHP" },
    { "python", "Python" },
    { "sh", "Shell" },
    { "tcl", "Tcl" },
    { "vala", "Vala" }
  };

  g_autoptr(GError) error = NULL;

  if (!dex_await (FOUNDRY_DIAGNOSTIC_TOOL_CLASS (plugin_gettext_diagnostic_tool_parent_class)->prepare (tool, launcher, argv, environ, language), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  for (guint i = 0; i < G_N_ELEMENTS (id_to_lang); i++)
    {
      if (g_strcmp0 (language, id_to_lang[i].id) == 0)
        {
          foundry_process_launcher_append_argv (launcher, "-L");
          foundry_process_launcher_append_argv (launcher, id_to_lang[i].lang);
          foundry_process_launcher_append_argv (launcher, "-o");
          foundry_process_launcher_append_argv (launcher, "-");
          foundry_process_launcher_append_argv (launcher, "-");

          return dex_future_new_true ();
        }
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Language %s is not supported",
                                language);
}

static void
plugin_gettext_diagnostic_tool_class_init (PluginGettextDiagnosticToolClass *klass)
{
  FoundryDiagnosticToolClass *diagnostic_tool_class = FOUNDRY_DIAGNOSTIC_TOOL_CLASS (klass);

  diagnostic_tool_class->dup_bytes_for_stdin = plugin_gettext_diagnostic_tool_dup_bytes_for_stdin;
  diagnostic_tool_class->extract_from_stdout = plugin_gettext_diagnostic_tool_extract_from_stdout;
  diagnostic_tool_class->prepare = plugin_gettext_diagnostic_tool_prepare;
}

static void
plugin_gettext_diagnostic_tool_init (PluginGettextDiagnosticTool *self)
{
  foundry_diagnostic_tool_set_argv (FOUNDRY_DIAGNOSTIC_TOOL (self),
                                    FOUNDRY_STRV_INIT ("xgettext",
                                                       "--check=ellipsis-unicode",
                                                       "--check=quote-unicode",
                                                       "--check=space-ellipsis",
                                                       "--from-code=UTF-8",
                                                       "-k_",
                                                       "-kN_"));
}
