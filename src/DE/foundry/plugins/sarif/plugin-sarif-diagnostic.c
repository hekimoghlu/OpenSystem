/* plugin-sarif-diagnostic.c
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

#include "plugin-sarif-diagnostic.h"

#if 0
{
  "ruleId" : "error",
  "level" : "error",
  "message" : {
    "text" : "expected ‘;’ before ‘}}’ token"
  },
  "locations" : [
    {
      "physicalLocation" : {
        "artifactLocation" : {
          "uri" : "../../src/gcc/testsuite/gcc.dg/missing-symbol-2.c",
          "uriBaseId" : "PWD"
        },
        "region" : {
          "startLine" : 35,
          "startColumn" : 12,
          "endColumn" : 13
        },
        "contextRegion" : {
          "startLine" : 35,
          "snippet" : {
            "text" : "  return 42 /* { dg-error \"expected ';'\" } */\n"
          }
        }
      },
      "logicalLocations" : [
        {
          "index" : 2,
          "fullyQualifiedName" : "missing_semicolon"
        }
      ]
    }
  ],
  "fixes" : [
    {
      "artifactChanges" : [
        {
          "artifactLocation" : {
            "uri" : "../../src/gcc/testsuite/gcc.dg/missing-symbol-2.c",
            "uriBaseId" : "PWD"
          },
          "replacements" : [
            {
              "deletedRegion" : {
                "startLine" : 35,
                "startColumn" : 12,
                "endColumn" : 12
              },
              "insertedContent" : {
                "text" : ";"
              }
            }
          ]
        }
      ]
    }
  ]
}
#endif

static GFile *
create_file (const char *uri,
             const char *uri_base_id,
             const char *builddir)
{
  /* Assume builddir for "PWD", but really we don't ever want to
   * get these and we should encourage GCC to send full paths or
   * URIs to the file.
   */

  if (foundry_str_equal0 (uri_base_id, "PWD"))
    {
      if (builddir != NULL)
        return g_file_new_build_filename (builddir, uri, NULL);
      else
        return g_file_new_build_filename (g_get_current_dir (), uri, NULL);
    }

  return g_file_new_for_uri (uri);
}

/* This doesn't currently handle everything SARIF can do, but we
 * can certainly extend our diagnostic API to support more.
 *
 * Especially since our 1.0 doesn't have "fixit" support natively
 * and would need to be applied via "code actions".
 */

FoundryDiagnostic *
plugin_sarif_diagnostic_new (FoundryContext *context,
                             JsonNode       *result,
                             const char     *builddir)
{
  g_autoptr(FoundryDiagnosticBuilder) builder = NULL;
  JsonNode *locations = NULL;
  JsonNode *fixes = NULL;
  const char *level = NULL;
  const char *text = NULL;
  const char *rule_id = NULL;
  const char *snippet_text = NULL;
  gint64 start_line = 0;
  gint64 start_column = 0;
  gint64 end_column = 0;
  gint64 context_start_line = 0;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (result != NULL, NULL);

  builder = foundry_diagnostic_builder_new (context);

  if (FOUNDRY_JSON_OBJECT_PARSE (result, "level", FOUNDRY_JSON_NODE_GET_STRING (&level)) && level)
    {
      if (g_str_equal (level, "error"))
        foundry_diagnostic_builder_set_severity (builder, FOUNDRY_DIAGNOSTIC_ERROR);
      else if (g_str_equal (level, "warning"))
        foundry_diagnostic_builder_set_severity (builder, FOUNDRY_DIAGNOSTIC_WARNING);
      else if (g_str_equal (level, "fatal"))
        foundry_diagnostic_builder_set_severity (builder, FOUNDRY_DIAGNOSTIC_FATAL);
      else if (g_str_equal (level, "note"))
        foundry_diagnostic_builder_set_severity (builder, FOUNDRY_DIAGNOSTIC_NOTE);
      else
        foundry_diagnostic_builder_set_severity (builder, FOUNDRY_DIAGNOSTIC_IGNORED);
    }

  if (FOUNDRY_JSON_OBJECT_PARSE (result, "ruleId", FOUNDRY_JSON_NODE_GET_STRING (&rule_id)))
    foundry_diagnostic_builder_set_rule_id (builder, rule_id);

  if (FOUNDRY_JSON_OBJECT_PARSE (result,
                                 "message", "{",
                                   "text", FOUNDRY_JSON_NODE_GET_STRING (&text),
                                 "}"))
    foundry_diagnostic_builder_set_message (builder, text);

  if (FOUNDRY_JSON_OBJECT_PARSE (result, "locations", FOUNDRY_JSON_NODE_GET_NODE (&locations)) &&
      JSON_NODE_HOLDS_ARRAY (locations))
    {
      JsonArray *ar = json_node_get_array (locations);
      guint length = json_array_get_length (ar);

      for (guint i = 0; i < length; i++)
        {
          JsonNode *location = json_array_get_element (ar, i);
          const char *uri = NULL;
          const char *uri_base_id = NULL;

          if (!FOUNDRY_JSON_OBJECT_PARSE (location,
                                          "physicalLocation", "{",
                                            "artifactLocation", "{",
                                             "uriBaseId", FOUNDRY_JSON_NODE_GET_STRING (&uri_base_id),
                                            "}",
                                          "}"))
            uri_base_id = NULL;

          if (FOUNDRY_JSON_OBJECT_PARSE (location,
                                         "physicalLocation", "{",
                                           "artifactLocation", "{",
                                             "uri", FOUNDRY_JSON_NODE_GET_STRING (&uri),
                                           "}",
                                           "region", "{",
                                             "startLine", FOUNDRY_JSON_NODE_GET_INT (&start_line),
                                             "startColumn", FOUNDRY_JSON_NODE_GET_INT (&start_column),
                                             "endColumn", FOUNDRY_JSON_NODE_GET_INT (&end_column),
                                           "}",
                                           "contextRegion", "{",
                                             "startLine", FOUNDRY_JSON_NODE_GET_INT (&context_start_line),
                                             "snippet", "{",
                                               "text", FOUNDRY_JSON_NODE_GET_STRING (&snippet_text),
                                             "}",
                                           "}",
                                         "}"))
            {
              if (i == 0)
                {
                  g_autoptr(GFile) file = create_file (uri, uri_base_id, builddir);

                  foundry_diagnostic_builder_set_line (builder, MAX (0, start_line - 1));
                  foundry_diagnostic_builder_set_line_offset (builder, MAX (0, start_column - 1));
                  foundry_diagnostic_builder_set_file (builder, file);
                }

              foundry_diagnostic_builder_add_range (builder,
                                                    MAX (0, start_line - 1),
                                                    MAX (0, start_column - 1),
                                                    MAX (0, start_line),
                                                    MAX (0, end_column - 1));
            }
        }
    }

  if (FOUNDRY_JSON_OBJECT_PARSE (result, "fixes", FOUNDRY_JSON_NODE_GET_NODE (&fixes)) &&
      JSON_NODE_HOLDS_ARRAY (fixes))
    {
      JsonArray *fixes_ar = json_node_get_array (fixes);
      guint n_fixes = json_array_get_length (fixes_ar);

      for (guint f = 0; f < n_fixes; f++)
        {
          JsonNode *fix = json_array_get_element (fixes_ar, f);
          JsonNode *replacements = NULL;
          JsonNode *changes = NULL;
          JsonArray *changes_ar;
          const char *description = NULL;
          const char *uri = NULL;
          const char *uri_base_id = NULL;
          guint n_changes;

          if (!FOUNDRY_JSON_OBJECT_PARSE (fix,
                                          "description", "{",
                                            "text", FOUNDRY_JSON_NODE_GET_STRING (&description),
                                          "}"))
            description = NULL;

          if (!FOUNDRY_JSON_OBJECT_PARSE (fix, "artifactChanges", FOUNDRY_JSON_NODE_GET_NODE (&changes)) ||
              !JSON_NODE_HOLDS_ARRAY (changes))
            continue;

          changes_ar = json_node_get_array (changes);
          n_changes = json_array_get_length (changes_ar);

          for (guint c = 0; c < n_changes; c++)
            {
              JsonNode *change = json_array_get_element (changes_ar, c);

              if (!FOUNDRY_JSON_OBJECT_PARSE (change,
                                              "artifactLocation", "{",
                                                "uriBaseId", FOUNDRY_JSON_NODE_GET_STRING (&uri_base_id),
                                              "}"))
                uri_base_id = NULL;

              if (FOUNDRY_JSON_OBJECT_PARSE (change,
                                             "artifactLocation", "{",
                                               "uri", FOUNDRY_JSON_NODE_GET_STRING (&uri),
                                             "}",
                                             "replacements", FOUNDRY_JSON_NODE_GET_NODE (&replacements)))
                {
                  JsonArray *replacements_ar = json_node_get_array (replacements);
                  g_autoptr(GListStore) edits = g_list_store_new (FOUNDRY_TYPE_TEXT_EDIT);
                  guint n_replacements = json_array_get_length (replacements_ar);

                  for (guint j = 0; j < n_replacements; j++)
                    {
                      JsonNode *replacement = json_array_get_element (replacements_ar, j);
                      const char *insert_text = NULL;

                      if (FOUNDRY_JSON_OBJECT_PARSE (replacement,
                                                     "deletedRegion", "{",
                                                       "startLine", FOUNDRY_JSON_NODE_GET_INT (&start_line),
                                                       "startColumn", FOUNDRY_JSON_NODE_GET_INT (&start_column),
                                                       "endColumn", FOUNDRY_JSON_NODE_GET_INT (&end_column),
                                                     "}",
                                                     "insertedContent", "{",
                                                       "text", FOUNDRY_JSON_NODE_GET_STRING (&insert_text),
                                                     "}"))
                        {
                          g_autoptr(GFile) file = create_file (uri, uri_base_id, builddir);
                          g_autoptr(FoundryTextEdit) text_edit = NULL;

                          text_edit = foundry_text_edit_new (file,
                                                             CLAMP (start_line, 1, G_MAXUINT) - 1,
                                                             CLAMP (start_column, 1, G_MAXUINT) - 1,
                                                             CLAMP (start_line, 1, G_MAXUINT) - 1,
                                                             CLAMP (end_column, 1, G_MAXUINT) - 1,
                                                             insert_text);
                          g_list_store_append (edits, text_edit);
                        }
                    }

                  foundry_diagnostic_builder_add_fix (builder, description, G_LIST_MODEL (edits));
                }
            }
        }
    }

  return foundry_diagnostic_builder_end (builder);
}
