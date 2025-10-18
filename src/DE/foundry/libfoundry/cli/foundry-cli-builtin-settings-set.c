/* foundry-cli-builtin-settings-set.c
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

#include "foundry-cli-builtin-private.h"
#include "foundry-context.h"
#include "foundry-settings.h"
#include "foundry-service.h"
#include "foundry-util-private.h"

static void
get_schemes (GStrvBuilder *builder,
             const char   *prefix)
{
  GSettingsSchemaSource *source = g_settings_schema_source_get_default ();
  g_auto(GStrv) non_relocatable = NULL;
  g_auto(GStrv) relocatable = NULL;

  g_settings_schema_source_list_schemas (source, TRUE, &non_relocatable, &relocatable);

  for (guint i = 0; non_relocatable[i]; i++)
    {
      if (g_str_has_prefix (non_relocatable[i], "app.devsuite.foundry."))
        {
          const char *suffix = non_relocatable[i] + strlen ("app.devsuite.foundry.");

          if (prefix == NULL || g_str_has_prefix (suffix, prefix))
            {
              g_autofree char *spaced = g_strdup_printf ("%s ", suffix);
              g_strv_builder_add (builder, spaced);
            }
        }
    }

  for (guint i = 0; relocatable[i]; i++)
    {
      if (g_str_has_prefix (relocatable[i], "app.devsuite.foundry."))
        {
          const char *suffix = relocatable[i] + strlen ("app.devsuite.foundry.");

          if (prefix == NULL || g_str_has_prefix (suffix, prefix))
            {
              g_autofree char *pathed = g_strdup_printf ("%s:/", suffix);
              g_strv_builder_add (builder, pathed);
            }
        }
    }
}

static void
get_keys (GStrvBuilder *builder,
          const char   *schema_id,
          const char   *current)
{
  GSettingsSchemaSource *source = g_settings_schema_source_get_default ();
  g_autoptr(GSettingsSchema) schema = NULL;
  g_auto(GStrv) keys = NULL;

  if (!(schema = g_settings_schema_source_lookup (source, schema_id, TRUE)))
    return;

  if ((keys = g_settings_schema_list_keys (schema)))
    {
      for (guint i = 0; keys[i]; i++)
        {
          if (current == NULL || g_str_has_prefix (keys[i], current))
            {
              g_autofree char *spaced = g_strdup_printf ("%s ", keys[i]);
              g_strv_builder_add (builder, spaced);
            }
        }
    }
}

static char **
foundry_cli_builtin_settings_set_complete (FoundryCommandLine *command_line,
                                           const char         *command,
                                           const GOptionEntry *entry,
                                           FoundryCliOptions  *options,
                                           const char * const *argv,
                                           const char         *current)
{
  g_autoptr(GStrvBuilder) builder = g_strv_builder_new ();
  int argc = g_strv_length ((char **)argv);

  if (argc == 1)
    {
      if (foundry_str_empty0 (current))
        get_schemes (builder, NULL);
    }
  else if (argc == 2)
    {
      if (current != NULL && g_str_equal (current, argv[1]))
        {
          get_schemes (builder, current);
        }
      else
        {
          g_autofree char *schema_id = g_strdup_printf ("app.devsuite.foundry.%s", argv[1]);
          get_keys (builder, schema_id, NULL);
        }
    }
  else if (argc == 3)
    {
      g_autofree char *schema_id = g_strdup_printf ("app.devsuite.foundry.%s", argv[1]);

      if (current != NULL && g_str_equal (current, argv[2]))
        get_keys (builder, schema_id, current);
    }

  return g_strv_builder_end (builder);
}

static void
foundry_cli_builtin_settings_set_help (FoundryCommandLine *command_line)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));

  foundry_command_line_print (command_line, "Usage:\n");
  foundry_command_line_print (command_line, "  foundry settings get schema key\n");
  foundry_command_line_print (command_line, "\n");
  foundry_command_line_print (command_line, "Options:\n");
  foundry_command_line_print (command_line, "  --help                Show help options\n");
  foundry_command_line_print (command_line, "\n");
}

static int
foundry_cli_builtin_settings_set_run (FoundryCommandLine *command_line,
                                      const char * const *argv,
                                      FoundryCliOptions  *options,
                                      DexCancellable     *cancellable)
{
  g_autoptr(GSettingsSchemaKey) _key = NULL;
  g_autoptr(FoundrySettings) settings = NULL;
  g_autoptr(GSettingsSchema) _schema = NULL;
  g_autoptr(FoundryContext) foundry = NULL;
  g_autoptr(GSettings) global_layer = NULL;
  g_autoptr(GSettings) project_layer = NULL;
  g_autoptr(GVariant) variant = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *schema = NULL;
  g_autofree char *key = NULL;
  g_autofree char *value_string = NULL;
  g_autofree char *schema_id = NULL;
  g_autofree char *path = NULL;
  const GVariantType *variant_type;

  gboolean global = FALSE;
  gboolean project = FALSE;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));
  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);
  g_assert (!cancellable || DEX_IS_CANCELLABLE (cancellable));

  if (foundry_cli_options_help (options))
    {
      foundry_cli_builtin_settings_set_help (command_line);
      return EXIT_SUCCESS;
    }

  if (argv[1] == NULL || argv[2] == NULL || argv[3] == NULL)
    {
      foundry_command_line_printerr (command_line, "usage: foundry settings get SCHEMA KEY VALUE\n");
      return EXIT_FAILURE;
    }

  if (!foundry_cli_options_get_boolean (options, "global", &global))
    global = FALSE;

  if (!foundry_cli_options_get_boolean (options, "project", &project))
    project = FALSE;

  if (global && project)
    {
      foundry_command_line_printerr (command_line, "--global and --project cannot both be specified\n");
      return EXIT_FAILURE;
    }

  if (strchr (argv[1], ':'))
    {
      const char *colon = strchr (argv[1], ':');

      path = g_strdup (colon + 1);
      schema_id = g_strndup (argv[1], colon - argv[1]);
    }
  else
    {
      schema_id = g_strdup (argv[1]);
    }

  schema = g_strdup_printf ("app.devsuite.foundry.%s", schema_id);
  key = g_strdup (argv[2]);

  if (!(_schema = g_settings_schema_source_lookup (g_settings_schema_source_get_default (),
                                                   schema, TRUE)))
    {
      foundry_command_line_printerr (command_line, "No such schema \"%s\"\n", schema);
      return EXIT_FAILURE;
    }

  if (!g_settings_schema_has_key (_schema, key))
    {
      foundry_command_line_printerr (command_line, "No such key \"%s\" in schema \"%s\"\n",
                                     key, schema);
      return EXIT_FAILURE;
    }

  _key = g_settings_schema_get_key (_schema, key);
  variant_type = g_settings_schema_key_get_value_type (_key);

  if (!(variant = g_variant_parse (variant_type, argv[3], NULL, NULL, &error)))
    {
      if (!g_variant_type_equal (variant_type, G_VARIANT_TYPE_STRING))
        {
          foundry_command_line_printerr (command_line, "Cannot parse value: %s\n",
                                         error->message);
          return EXIT_FAILURE;
        }

      variant = g_variant_take_ref (g_variant_new_string (argv[3]));
    }

  if (!(foundry = dex_await_object (foundry_cli_options_load_context (options, command_line), &error)))
    goto handle_error;

  settings = foundry_context_load_settings (foundry, schema, path);
  global_layer = foundry_settings_dup_layer (settings, FOUNDRY_SETTINGS_LAYER_APPLICATION);
  project_layer = foundry_settings_dup_layer (settings, FOUNDRY_SETTINGS_LAYER_PROJECT);

  if (global)
    g_settings_set_value (global_layer, key, variant);
  else if (project)
    g_settings_set_value (project_layer, key, variant);
  else
    foundry_settings_set_value (settings, key, variant);

  return EXIT_SUCCESS;

handle_error:

  foundry_command_line_printerr (command_line, "%s\n", error->message);
  return EXIT_FAILURE;
}

void
foundry_cli_builtin_settings_set (FoundryCliCommandTree *tree)
{
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "settings", "set"),
                                     &(FoundryCliCommand) {
                                       .options = (GOptionEntry[]) {
                                         { "global", 'g', 0, G_OPTION_ARG_NONE },
                                         { "project", 'p', 0, G_OPTION_ARG_NONE },
                                         { "help", 0, 0, G_OPTION_ARG_NONE },
                                         {0}
                                       },
                                       .run = foundry_cli_builtin_settings_set_run,
                                       .prepare = NULL,
                                       .complete = foundry_cli_builtin_settings_set_complete,
                                       .gettext_package = GETTEXT_PACKAGE,
                                       .description = N_("SCCHEMA KEY - Get setting"),
                                     });
}
