/* plugin-gdiagnose-diagnostic-provider.c
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

#include "line-reader-private.h"

#include "plugin-gdiagnose-diagnostic-provider.h"

struct _PluginGdiagnoseDiagnosticProvider
{
  FoundryDiagnosticProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginGdiagnoseDiagnosticProvider, plugin_gdiagnose_diagnostic_provider, FOUNDRY_TYPE_DIAGNOSTIC_PROVIDER)

static char *
get_word (const char *ptr,
          const char *endptr)
{
  const char *wordend = ptr;
  gunichar ch;

  while (wordend < endptr &&
         (ch = g_utf8_get_char (wordend)) &&
         (ch == '_' || g_unichar_isalnum (ch)))
    wordend = g_utf8_next_char (wordend);

  return g_strndup (ptr, wordend - ptr);
}

static DexFuture *
plugin_gdiagnose_diagnostic_provider_diagnose_fiber (FoundryContext *context,
                                                     GFile          *file,
                                                     GBytes         *contents)
{
  g_autoptr(FoundryDiagnosticBuilder) builder = NULL;
  g_autoptr(GListStore) store = NULL;
  const char *function_name_line = NULL;
  const char *data;
  const char *line;
  const char *endptr;
  LineReader reader;
  gsize line_len;
  gsize len;
  guint lineno = 0;

  g_assert (contents != NULL);
  g_assert (!file || G_IS_FILE (file));

  /* TODO: Once librig is ready, we should use that instead of
   *       this code as that will give us AST insight into what
   *       is be called and where. This is very rudimentary but
   *       still extremely useful.
   */

  data = g_bytes_get_data (contents, &len);
  endptr = data + len;

  if (!g_utf8_validate_len (data, len, NULL))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_INVALID_DATA,
                                  "Contents must be UTF-8");

  store = g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC);
  builder = foundry_diagnostic_builder_new (context);

  if (file)
    foundry_diagnostic_builder_set_file (builder, file);

  line_reader_init (&reader, (char *)data, len);

  while ((line = line_reader_next (&reader, &line_len)))
    {
      const char *match;

      lineno++;

      if (line[0] == '_' || g_unichar_isalpha (g_utf8_get_char (line)))
        {
          function_name_line = line;
          continue;
        }

      if (function_name_line == NULL)
        continue;

      if (line[0] == '}')
        {
          function_name_line = NULL;
          continue;
        }

      if ((match = memmem (line, line_len, "parent_class)->", 15)))
        {
          const char *begin = match + 15;
          g_autofree char *chainup = get_word (begin, endptr);
          g_autofree char *function = get_word (function_name_line, endptr);

          if (function == NULL || chainup == NULL)
            continue;

          if (!g_str_has_suffix (function, chainup))
            {
              g_autoptr(FoundryDiagnostic) diagnostic = NULL;

              foundry_diagnostic_builder_set_line (builder, lineno);
              foundry_diagnostic_builder_set_line_offset (builder, 1 + g_utf8_strlen (line, begin - line));
              foundry_diagnostic_builder_take_message (builder,
                                                       g_strdup_printf ("Possibly incorrect chain-up to '%s' from '%s'",
                                                                        chainup, function));

              diagnostic = foundry_diagnostic_builder_end (builder);

              g_list_store_append (store, diagnostic);
            }
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
plugin_gdiagnose_diagnostic_provider_diagnose (FoundryDiagnosticProvider *self,
                                               GFile                     *file,
                                               GBytes                    *contents,
                                               const char                *language)
{
  g_autoptr(FoundryContext) context = NULL;

  g_assert (PLUGIN_IS_GDIAGNOSE_DIAGNOSTIC_PROVIDER (self));
  g_assert (!file || G_IS_FILE (file));
  g_assert (file || contents);

  if (contents == NULL || !g_strv_contains (FOUNDRY_STRV_INIT ("c", "chdr", "cpp", "cpphdr", "objc"), language))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_SUPPORTED,
                                  "Not supported");

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  return foundry_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                  G_CALLBACK (plugin_gdiagnose_diagnostic_provider_diagnose_fiber),
                                  3,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  G_TYPE_FILE, file,
                                  G_TYPE_BYTES, contents);
}

static void
plugin_gdiagnose_diagnostic_provider_class_init (PluginGdiagnoseDiagnosticProviderClass *klass)
{
  FoundryDiagnosticProviderClass *diagnostic_provider_class = FOUNDRY_DIAGNOSTIC_PROVIDER_CLASS (klass);

  diagnostic_provider_class->diagnose = plugin_gdiagnose_diagnostic_provider_diagnose;
}

static void
plugin_gdiagnose_diagnostic_provider_init (PluginGdiagnoseDiagnosticProvider *self)
{
}
