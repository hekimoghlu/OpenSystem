/* foundry-command-line.c
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

#include <glib/gstdio.h>
#include <glib/gi18n-lib.h>

#include <json-glib/json-glib.h>

#include "foundry-cli-command-private.h"
#include "foundry-cli-command-tree.h"
#include "foundry-command-line-private.h"
#include "foundry-command-line-input-private.h"
#include "foundry-command-line-local-private.h"
#include "foundry-command-line-remote-private.h"
#include "foundry-ipc.h"
#include "foundry-tty-auth-provider.h"
#include "foundry-util-private.h"

G_DEFINE_ABSTRACT_TYPE (FoundryCommandLine, foundry_command_line, G_TYPE_OBJECT)
G_DEFINE_QUARK (foundry_command_line_error, foundry_command_line_error)
G_DEFINE_ENUM_TYPE (FoundryObjectSerializerFormat, foundry_object_serializer_format,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_OBJECT_SERIALIZER_FORMAT_TEXT, "text"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_OBJECT_SERIALIZER_FORMAT_JSON, "json"))

static DexCancellable *
foundry_command_line_dup_cancellable (FoundryCommandLine *self)
{
  g_assert (FOUNDRY_IS_COMMAND_LINE (self));

  if (FOUNDRY_COMMAND_LINE_GET_CLASS (self)->dup_cancellable)
    return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->dup_cancellable (self);

  return dex_cancellable_new ();
}

static DexFuture *
foundry_command_line_real_run (FoundryCommandLine *self,
                               const char * const *argv)
{
  g_autoptr(FoundryCliOptions) options = NULL;
  g_autoptr(DexCancellable) cancellable = NULL;
  g_autoptr(GOptionContext) context = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree GType *types = NULL;
  g_autofree char *foundry_dir = NULL;
  g_autofree char *command_name = NULL;
  const FoundryCliCommand *command;
  FoundryCliCommandTree *tree;
  g_auto(GStrv) args = NULL;
  int argc;

  const GOptionEntry global_entries[] = {
    { "foundry-dir", 0, 0, G_OPTION_ARG_FILENAME, &foundry_dir, N_("Set the path to the .foundry dir"), N_("DIR") },
    { 0 }
  };

  g_assert (FOUNDRY_IS_COMMAND_LINE (self));
  g_assert (argv != NULL);

  if (argv[0] == NULL || argv[1] == NULL)
    {
      foundry_command_line_help (self);
      return dex_future_new_for_int (EXIT_FAILURE);
    }

  cancellable = foundry_command_line_dup_cancellable (self);

  args = g_strdupv ((char **)argv);
  argc = g_strv_length (args);

  context = g_option_context_new (NULL);
  g_option_context_add_main_entries (context, global_entries, GETTEXT_PACKAGE);
  g_option_context_set_ignore_unknown_options (context, TRUE);
  g_option_context_set_help_enabled (context, FALSE);
  g_option_context_set_strict_posix (context, TRUE);

  if (!g_option_context_parse (context, &argc, &args, &error))
    {
      foundry_command_line_printerr (self, "%s: %s\n", _("error"), error->message);
      return dex_future_new_for_int (EXIT_FAILURE);
    }

  if (args[0] == NULL || args[1] == NULL)
    {
      foundry_command_line_help (self);
      return dex_future_new_for_int (EXIT_FAILURE);
    }

  command_name = g_strdup (args[1]);

  if (g_str_equal (command_name, "help") || g_str_equal (command_name, "--help"))
    {
      foundry_command_line_help (self);
      return dex_future_new_for_int (EXIT_SUCCESS);
    }

  tree = foundry_cli_command_tree_get_default ();

  if (!(command = foundry_cli_command_tree_lookup (tree, &args, &options, &error)))
    {
      foundry_command_line_printerr (self, "%s: %s\n", _("error"), error->message);
      return dex_future_new_for_int (EXIT_FAILURE);
    }

  if (foundry_dir != NULL)
    foundry_cli_options_set_string (options, "foundry-dir", foundry_dir);

  return foundry_cli_command_run (command,
                                  self,
                                  (const char * const *)args,
                                  options,
                                  cancellable);
}

static void
foundry_command_line_class_init (FoundryCommandLineClass *klass)
{
  klass->run = foundry_command_line_real_run;

  g_dbus_error_register_error (FOUNDRY_COMMAND_LINE_ERROR,
                               FOUNDRY_COMMAND_LINE_ERROR_RUN_LOCAL,
                               "app.devsuite.foundry.CommandLine.Error.RunLocal");
}

static void
foundry_command_line_init (FoundryCommandLine *self)
{
}

FoundryCommandLine *
foundry_command_line_new (void)
{
  return foundry_command_line_local_new ();
}

void
foundry_command_line_print (FoundryCommandLine *self,
                            const char         *format,
                            ...)
{
  g_autofree char *message = NULL;
  va_list args;

  g_return_if_fail (FOUNDRY_IS_COMMAND_LINE (self));
  g_return_if_fail (format != NULL);

  va_start (args, format);
  message = g_strdup_vprintf (format, args);
  va_end (args);

  FOUNDRY_COMMAND_LINE_GET_CLASS (self)->print (self, message);
}

void
foundry_command_line_printerr (FoundryCommandLine *self,
                               const char         *format,
                               ...)
{
  g_autofree char *message = NULL;
  va_list args;

  g_return_if_fail (FOUNDRY_IS_COMMAND_LINE (self));
  g_return_if_fail (format != NULL);

  va_start (args, format);
  message = g_strdup_vprintf (format, args);
  va_end (args);

  FOUNDRY_COMMAND_LINE_GET_CLASS (self)->printerr (self, message);
}

gboolean
foundry_command_line_isatty (FoundryCommandLine *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), FALSE);

  return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->isatty (self);
}

void
foundry_command_line_help (FoundryCommandLine *self)
{
  g_return_if_fail (FOUNDRY_IS_COMMAND_LINE (self));

  foundry_command_line_print (self, "%s\n", _("Usage:"));
  foundry_command_line_print (self, "  foundry [OPTIONS…] COMMAND\n");
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("Commands"));
  foundry_command_line_print (self, "  init                 %s\n", _("Initialize a new project in current directory"));
  //foundry_command_line_print (self, "  create               %s\n", _("Create a new project from template"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("Environment Commands"));
  foundry_command_line_print (self, "  enter                %s\n", _("Enter environment of project in current directory"));
  foundry_command_line_print (self, "  devenv               %s\n", _("Enter build environment for project"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("Build Commands"));
  foundry_command_line_print (self, "  build                %s\n", _("Build the current build pipeline"));
  foundry_command_line_print (self, "  rebuild              %s\n", _("Wipe and rebuild the current build pipeline"));
  foundry_command_line_print (self, "  clean                %s\n", _("Clean the current build pipeline"));
  foundry_command_line_print (self, "  install              %s\n", _("Run the install target for the project"));
  foundry_command_line_print (self, "  deploy               %s\n", _("Deploy the project to the active device"));
  foundry_command_line_print (self, "  export               %s\n", _("Export project artifacts"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("Run Commands"));
  foundry_command_line_print (self, "  run                  %s\n", _("Run a project command"));
  //foundry_command_line_print (self, "  debug                %s\n", _("Run a project commmnd in the debugger"));
  //foundry_command_line_print (self, "  profile              %s\n", _("Run a project commmnd in the profiler"));
  //foundry_command_line_print (self, "  valgrind             %s\n", _("Run a project commmnd under valgrind"));
  //foundry_command_line_print (self, "  test                 %s\n", _("Run unit tests"));
  foundry_command_line_print (self, "\n");
  //foundry_command_line_print (self, "%s:\n", _("User Commands"));
  //foundry_command_line_print (self, "  command list         %s\n", _("List registered commands"));
  //foundry_command_line_print (self, "  command add          %s\n", _("Add a new user command or group"));
  //foundry_command_line_print (self, "  command remove       %s\n", _("Remove a user command or group"));
  //foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("File/Directory Commands"));
  //foundry_command_line_print (self, "  search               %s\n", _("Search the project"));
  //foundry_command_line_print (self, "  replace              %s\n", _("Search and replace within the project"));
  //foundry_command_line_print (self, "  index                %s\n", _("Update source code indexes"));
  //foundry_command_line_print (self, "  format               %s\n", _("Reformat a file"));
  //foundry_command_line_print (self, "  symbols              %s\n", _("List symbols within a file"));
  foundry_command_line_print (self, "  diagnose             %s\n", _("List diagnostics within a file or files"));
  foundry_command_line_print (self, "  show                 %s\n", _("Open a file or files in file browser"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("Documentation Commands"));
  foundry_command_line_print (self, "  doc bundle list      %s\n", _("List available documentation bundles"));
  foundry_command_line_print (self, "  doc bundle install   %s\n", _("Install a specific documentation bundle"));
  foundry_command_line_print (self, "  doc query            %s\n", _("Search for documentation"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("Configuration Commands"));
  foundry_command_line_print (self, "  config list          %s\n", _("List available configurations"));
  foundry_command_line_print (self, "  config switch        %s\n", _("Change the active configuration"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("SDK Commands"));
  foundry_command_line_print (self, "  sdk list             %s\n", _("List available SDKs"));
  foundry_command_line_print (self, "  sdk install          %s\n", _("Install a specific SDK"));
  foundry_command_line_print (self, "  sdk switch           %s\n", _("Change the active SDK for the build pipeline"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("Pipeline Commands"));
  foundry_command_line_print (self, "  pipeline info        %s\n", _("Show information about the pipeline"));
  foundry_command_line_print (self, "  pipeline build       %s\n", _("Build the current pipeline"));
  foundry_command_line_print (self, "  pipeline rebuild     %s\n", _("Rebuild the current pipeline"));
  foundry_command_line_print (self, "  pipeline clean       %s\n", _("Clean the current pipeline"));
  foundry_command_line_print (self, "  pipeline purge       %s\n", _("Delete contents related to build"));
  foundry_command_line_print (self, "  pipeline configure   %s\n", _("Delete contents related to build"));
  //foundry_command_line_print (self, "  pipeline insert      %s\n", _("Insert a command into the pipeline"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("Device Commands"));
  foundry_command_line_print (self, "  device list          %s\n", _("List available devices"));
  foundry_command_line_print (self, "  device switch        %s\n", _("Switch the current target device"));
  //foundry_command_line_print (self, "  device pair              %s\n", _("Pair a device"));
  //foundry_command_line_print (self, "  device shell             %s\n", _("Open a shell on a device"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "%s:\n", _("LSP Commands"));
  foundry_command_line_print (self, "  lsp list             %s\n", _("List available language server plugins"));
  foundry_command_line_print (self, "  lsp run              %s\n", _("Run a language server for specific language"));
  foundry_command_line_print (self, "  lsp prefer           %s\n", _("Set preferred LSP for language"));
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "Examples:\n");
  foundry_command_line_print (self, "  # Enter project directory\n");
  foundry_command_line_print (self, "  cd ~/Projects/gnome-builder\n");
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "  # Run once to setup foundry\n");
  foundry_command_line_print (self, "  foundry init\n");
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "  # Build the project\n");
  foundry_command_line_print (self, "  foundry build\n");
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "  # Run the project\n");
  foundry_command_line_print (self, "  foundry run\n");
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "  # Enter the build environment\n");
  foundry_command_line_print (self, "  foundry devenv\n");
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "  # Start a shell in 'Run' environment\n");
  foundry_command_line_print (self, "  foundry run -- bash\n");
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "  # Start a language server on stdin/out for python\n");
  foundry_command_line_print (self, "  foundry lsp run python3\n");
  foundry_command_line_print (self, "\n");
  foundry_command_line_print (self, "  # Entry persistent foundry IDE environment\n");
  foundry_command_line_print (self, "  # where commands will run in parent process.\n");
  foundry_command_line_print (self, "  foundry enter\n");
  foundry_command_line_print (self, "\n");
}

/**
 * foundry_command_line_run:
 * @self: a #FoundryCommandLine
 *
 * Runs the command line.
 *
 * Returns: (transfer full): a #DexFuture that resolves to an int
 */
DexFuture *
foundry_command_line_run (FoundryCommandLine *self,
                          const char * const *argv)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), NULL);
  g_return_val_if_fail (argv != NULL, NULL);

  return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->run (self, argv);
}

char *
foundry_command_line_get_directory (FoundryCommandLine *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), NULL);

  return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->get_directory (self);
}

/**
 * foundry_command_line_get_environ:
 * @self: a #FoundryCommandLine
 *
 * Returns: (transfer full): the environment of the command line
 */
char **
foundry_command_line_get_environ (FoundryCommandLine *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), NULL);

  return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->get_environ (self);
}

gboolean
foundry_command_line_is_remote (FoundryCommandLine *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), FALSE);

  return FOUNDRY_IS_COMMAND_LINE_REMOTE (self);
}

DexFuture *
foundry_command_line_open (FoundryCommandLine *self,
                           int                 fd_number)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), NULL);

  if (fd_number < 0)
    return dex_future_new_reject (G_FILE_ERROR,
                                  G_FILE_ERROR_BADF,
                                  "Invalid fd number");

  return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->open (self, fd_number);
}

const char *
foundry_command_line_getenv (FoundryCommandLine *self,
                             const char         *name)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), NULL);

  return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->getenv (self, name);
}

int
foundry_command_line_get_stdin (FoundryCommandLine *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), -1);

  return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->get_stdin (self);
}

int
foundry_command_line_get_stdout (FoundryCommandLine *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), -1);

  return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->get_stdout (self);
}

int
foundry_command_line_get_stderr (FoundryCommandLine *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), -1);

  return FOUNDRY_COMMAND_LINE_GET_CLASS (self)->get_stderr (self);
}

void
foundry_command_line_print_object (FoundryCommandLine                 *self,
                                   GObject                            *object,
                                   const FoundryObjectSerializerEntry *entries,
                                   FoundryObjectSerializerFormat       format)
{
  g_autoptr(GListStore) store = NULL;

  g_return_if_fail (FOUNDRY_IS_COMMAND_LINE (self));
  g_return_if_fail (G_IS_OBJECT (object));

  store = g_list_store_new (G_OBJECT_TYPE (object));
  g_list_store_append (store, object);
  foundry_command_line_print_list (self, G_LIST_MODEL (store), entries, format, G_TYPE_INVALID);
}

typedef struct _Column
{
  const char *title;
  GParamSpec *pspec;
  gsize       longest;
  guint       is_boolean : 1;
  guint       is_number : 1;
  guint       is_enum : 1;
  guint       is_strv : 1;
} Column;

static void
foundry_command_line_print_sized (FoundryCommandLine *self,
                                  gsize               size,
                                  const char         *message,
                                  gboolean            bold)
{
  gsize len = 0;

  if (message != NULL)
    {
      len = strlen (message);

      if (bold && foundry_command_line_isatty (self))
        foundry_command_line_print (self, "\e[1m%s\e[22m", message);
      else
        foundry_command_line_print (self, "%s", message);
    }

  for (; len < size; len++)
    foundry_command_line_print (self, " ");
}

static gboolean
is_number_type (GType type)
{
  switch ((int) type)
    {
    case G_TYPE_UINT:
    case G_TYPE_UINT64:
    case G_TYPE_INT:
    case G_TYPE_INT64:
    case G_TYPE_LONG:
    case G_TYPE_ULONG:
    case G_TYPE_DOUBLE:
    case G_TYPE_FLOAT:
      return TRUE;

    default:
      return FALSE;
    }
}

void
foundry_command_line_print_list (FoundryCommandLine                 *self,
                                 GListModel                         *model,
                                 const FoundryObjectSerializerEntry *entries,
                                 FoundryObjectSerializerFormat       format,
                                 GType                               expected_type)
{
  g_autofree Column *columns = NULL;
  g_autoptr(GTypeClass) klass = NULL;
  g_autoptr(GStringChunk) chunk = NULL;
  g_autoptr(GPtrArray) strings = NULL;
  GType item_type;
  guint n_items;
  guint n_columns;

  g_assert (FOUNDRY_IS_COMMAND_LINE (self));
  g_assert (G_IS_LIST_MODEL (model));
  g_assert (entries != NULL);

  chunk = g_string_chunk_new (4096);
  strings = g_ptr_array_new ();

  if (expected_type != G_TYPE_INVALID)
    item_type = expected_type;
  else
    item_type = g_list_model_get_item_type (model);

  klass = g_type_class_ref (item_type);

  for (n_columns = 0; entries[n_columns].property; n_columns++);
  columns = g_new0 (Column, n_columns);

  for (guint c = 0; c < n_columns; c++)
    {
      columns[c].title = entries[c].heading;
      columns[c].longest = strlen (entries[c].heading);
      columns[c].pspec = g_object_class_find_property (G_OBJECT_CLASS (klass), entries[c].property);

      if (columns[c].pspec == NULL)
        {
          g_critical ("Object type %s does not have property '%s'",
                      g_type_name (item_type),
                      entries[c].property);
          return;
        }

      columns[c].is_enum = G_TYPE_IS_ENUM (columns[c].pspec->value_type);
      columns[c].is_strv = columns[c].pspec->value_type == G_TYPE_STRV;
      columns[c].is_boolean = columns[c].pspec->value_type == G_TYPE_BOOLEAN;
      columns[c].is_number = is_number_type (columns[c].pspec->value_type);
    }

  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(GObject) object = g_list_model_get_item (model, i);

      for (guint ii = 0; entries[ii].property; ii++)
        {
          Column *column = &columns[ii];
          g_auto(GValue) value = G_VALUE_INIT;
          const char *str;

          if (column->is_boolean)
            {
              g_value_init (&value, G_TYPE_BOOLEAN);
              g_object_get_property (object, entries[ii].property, &value);

              if (format == FOUNDRY_OBJECT_SERIALIZER_FORMAT_JSON)
                str = g_value_get_boolean (&value) ? "true" : "false";
              else
                {
                  if (g_value_get_boolean (&value))
                    str = g_string_chunk_insert_const (chunk, _("Yes"));
                  else
                    str = g_string_chunk_insert_const (chunk, _("No"));
                }
            }
          else if (column->is_enum)
            {
              g_autoptr(GEnumClass) enum_class = g_type_class_ref (column->pspec->value_type);
              GEnumValue *v;

              g_value_init (&value, column->pspec->value_type);
              g_object_get_property (object, entries[ii].property, &value);

              if ((v = g_enum_get_value (enum_class, g_value_get_enum (&value))))
                str = g_intern_string (v->value_nick);
              else
                str = NULL;
            }
          else if (column->is_strv)
            {
              g_auto(GStrv) strv = NULL;
              g_autoptr(GString) gstr = g_string_new (NULL);

              g_value_init (&value, G_TYPE_STRV);
              g_object_get_property (object, entries[ii].property, &value);

              if ((strv = g_value_dup_boxed (&value)))
                {
                  for (guint z = 0; strv[z]; z++)
                    {
                      g_autofree char *quoted = g_shell_quote (strv[z]);

                      if (gstr->len > 0)
                        g_string_append_c (gstr, ' ');
                      g_string_append (gstr, quoted);
                    }
                }

              g_value_unset (&value);
              g_value_init (&value, G_TYPE_STRING);
              g_value_take_string (&value, g_string_free (g_steal_pointer (&gstr), FALSE));

              str = g_value_get_string (&value);
            }
          else
            {
              g_value_init (&value, G_TYPE_STRING);
              g_object_get_property (object, entries[ii].property, &value);

              str = g_value_get_string (&value);
            }

          if (str != NULL)
            {
              g_autofree char *escaped = g_strescape (str, NULL);
              column->longest = MAX (column->longest, strlen (escaped));
              str = g_string_chunk_insert_const (chunk, escaped);
            }

          g_ptr_array_add (strings, (char *)str);
        }
    }

  if (format == FOUNDRY_OBJECT_SERIALIZER_FORMAT_TEXT)
    {
      for (guint c = 0; c < n_columns; c++)
        {
          const Column *column = &columns[c];
          guint len = c + 1 == n_columns ? strlen (column->title) : column->longest;

          if (c > 0)
            foundry_command_line_print (self, "  ");

          foundry_command_line_print_sized (self, len, column->title, TRUE);
        }

      foundry_command_line_print (self, "\n");

      g_assert (strings->len % n_columns == 0);

      for (guint i = 0; i < n_items; i++)
        {
          for (guint ii = 0; ii < n_columns; ii++)
            {
              const Column *column = &columns[ii];
              const char *str = strings->pdata[i * n_columns + ii];

              if (ii > 0)
                foundry_command_line_print (self, "  ");

              foundry_command_line_print_sized (self, column->longest, str, FALSE);
            }

          foundry_command_line_print (self, "\n");
        }
    }
  else if (format == FOUNDRY_OBJECT_SERIALIZER_FORMAT_JSON)
    {
      foundry_command_line_print (self, "[");

      for (guint i = 0; i < n_items; i++)
        {
          if (i > 0)
            foundry_command_line_print (self, ", ");

          foundry_command_line_print (self, "{");

          for (guint ii = 0; ii < n_columns; ii++)
            {
              const Column *column = &columns[ii];
              const char *str = strings->pdata[i * n_columns + ii];

              if (ii > 0)
                foundry_command_line_print (self, ", ");

              foundry_command_line_print (self, "\"%s\": ", column->pspec->name);

              if (column->is_boolean)
                foundry_command_line_print (self, "%s", str);
              else if (str == NULL)
                foundry_command_line_print (self, "null");
              else if (column->is_number)
                foundry_command_line_print (self, "%s", str);
              else
                foundry_command_line_print (self, "\"%s\"", str);
            }

          foundry_command_line_print (self, "}");
        }

      foundry_command_line_print (self, "]\n");
    }
}

FoundryObjectSerializerFormat
foundry_object_serializer_format_parse (const char *string)
{
  static GEnumClass *klass;
  GEnumValue *value;

  if (string == NULL)
    return FOUNDRY_OBJECT_SERIALIZER_FORMAT_TEXT;

  if G_UNLIKELY (klass == NULL)
    klass = g_type_class_ref (FOUNDRY_TYPE_OBJECT_SERIALIZER_FORMAT);

  if ((value = g_enum_get_value_by_nick (klass, string)))
    return value->value;

  return FOUNDRY_OBJECT_SERIALIZER_FORMAT_TEXT;
}

void
foundry_command_line_clear_progress (FoundryCommandLine *self)
{
  g_return_if_fail (FOUNDRY_IS_COMMAND_LINE (self));

  if (foundry_command_line_isatty (self))
    foundry_command_line_print (self, "\033]9;4;0\e\\");
}

void
foundry_command_line_set_progress (FoundryCommandLine *self,
                                   guint               percent)
{
  g_return_if_fail (FOUNDRY_IS_COMMAND_LINE (self));

  if (foundry_command_line_isatty (self))
    foundry_command_line_print (self, "\033]9;4;1;%d\e\\", MIN (percent, 100));
}

void
foundry_command_line_clear_line (FoundryCommandLine *self)
{
  g_return_if_fail (FOUNDRY_IS_COMMAND_LINE (self));

  if (foundry_command_line_isatty (self))
    foundry_command_line_print (self, "\033[2K\r");
  else
    foundry_command_line_print (self, "\n");
}

void
foundry_command_line_set_title (FoundryCommandLine *self,
                                const char         *title)
{
  g_autofree char *escaped = NULL;
  g_autofree char *command = NULL;
  gsize len;

  g_return_if_fail (FOUNDRY_IS_COMMAND_LINE (self));

  if (title == NULL)
    escaped = g_strdup ("");
  else
    escaped = g_strescape (title, NULL);

  command = g_strdup_printf ("\e]0;%s\e\\", escaped);
  len = strlen (command);

  if (len != write (foundry_command_line_get_stdout (self), command, len))
    {
      /* Do Nothing */
    }
}

/**
 * foundry_command_line_dup_auth_provider:
 * @self: a [class@Foundry.CommandLine]
 *
 * Gets an auth provider for this command line client if possible.
 *
 * Currrently, this only returns an auth provider if the command line
 * client provides a TTY.
 *
 * Returns: (transfer full) (nullable):
 */
FoundryAuthProvider *
foundry_command_line_dup_auth_provider (FoundryCommandLine *self)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), NULL);

  if (foundry_command_line_isatty (self))
    return foundry_tty_auth_provider_new (foundry_command_line_get_stdin (self));

  return NULL;
}

/**
 * foundry_command_line_build_file_for_arg:
 * @self: a [class@Foundry.CommandLine]
 *
 * Returns: (transfer full):
 */
GFile *
foundry_command_line_build_file_for_arg (FoundryCommandLine *self,
                                         const char         *arg)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (self), NULL);
  g_return_val_if_fail (arg != NULL, NULL);

  if (g_path_is_absolute (arg))
    return g_file_new_for_path (arg);

  return g_file_new_build_filename (foundry_command_line_get_directory (self), arg, NULL);
}

/**
 * foundry_command_line_request_input:
 * @self: a [class@Foundry.CommandLine]
 * @input: a [class@Foundry.Input]
 *
 * Queries the user for the information requested in @input.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 */
DexFuture *
foundry_command_line_request_input (FoundryCommandLine *self,
                                    FoundryInput       *input)
{
  int pty_fd;

  dex_return_error_if_fail (FOUNDRY_IS_COMMAND_LINE (self));
  dex_return_error_if_fail (FOUNDRY_IS_INPUT (input));

  pty_fd = foundry_command_line_get_stdout (self);

  return foundry_command_line_input (pty_fd, input);
}
