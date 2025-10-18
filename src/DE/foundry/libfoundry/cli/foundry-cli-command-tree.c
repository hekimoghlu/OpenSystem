/* foundry-cli-command-tree.c
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

#include "foundry-cli-command-private.h"
#include "foundry-cli-command-tree-private.h"

struct _FoundryCliCommandTree
{
  GObject  parent_instance;
  GNode   *root;
};

typedef struct _FoundryCliCommandTreeData
{
  char              *name;
  FoundryCliCommand *command;
} FoundryCliCommandTreeData;

G_DEFINE_FINAL_TYPE (FoundryCliCommandTree, foundry_cli_command_tree, G_TYPE_OBJECT)

static gboolean
is_null_or_empty (const char *str)
{
  return str == NULL || str[0] == 0;
}

static gboolean
has_prefix_or_equal (const char *str,
                     const char *prefix)
{
  return *prefix == 0 || g_str_has_prefix (str, prefix) || g_str_equal (str, prefix);
}

static void
free_data (FoundryCliCommandTreeData *data)
{
  g_clear_pointer (&data->name, g_free);
  g_clear_pointer (&data->command, foundry_cli_command_free);
  g_free (data);
}

static gboolean
free_command_traverse (GNode    *node,
                       gpointer  user_data)
{
  g_clear_pointer (&node->data, free_data);
  return FALSE;
}

static void
free_node (GNode *root)
{
  g_node_traverse (root, G_IN_ORDER, G_TRAVERSE_ALL, -1, free_command_traverse, NULL);
  g_node_destroy (root);
}

static void
foundry_cli_command_tree_finalize (GObject *object)
{
  FoundryCliCommandTree *self = (FoundryCliCommandTree *)object;

  g_clear_pointer (&self->root, free_node);

  G_OBJECT_CLASS (foundry_cli_command_tree_parent_class)->finalize (object);
}

static void
foundry_cli_command_tree_class_init (FoundryCliCommandTreeClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_cli_command_tree_finalize;
}

static void
foundry_cli_command_tree_init (FoundryCliCommandTree *self)
{
  self->root = g_node_new (NULL);
}

FoundryCliCommandTree *
foundry_cli_command_tree_new (void)
{
  return g_object_new (FOUNDRY_TYPE_CLI_COMMAND_TREE, NULL);
}

static void
insert_node (GNode                     *node,
             const char * const        *path,
             FoundryCliCommandTreeData *data)
{
  FoundryCliCommandTreeData *new_data;
  GNode *new_child;

  if (path[0] == NULL)
    {
      g_clear_pointer (&node->data, free_data);
      node->data = data;
      return;
    }

  g_assert (path[0] != NULL);

  for (GNode *child = node->children; child != NULL; child = child->next)
    {
      FoundryCliCommandTreeData *child_data = child->data;

      g_assert (child_data != NULL);

      if (g_str_equal (child_data->name, path[0]))
        {
          insert_node (child, &path[1], data);
          return;
        }
    }

  new_data = g_new0 (FoundryCliCommandTreeData, 1);
  new_data->name = g_strdup (path[0]);
  new_data->command = NULL;
  new_child = g_node_new (new_data);

  g_node_append (node, new_child);

  insert_node (new_child, &path[1], data);
}

void
foundry_cli_command_tree_register (FoundryCliCommandTree   *self,
                                   const char * const      *path,
                                   const FoundryCliCommand *command)
{
  FoundryCliCommandTreeData *data;
  gsize n_parts;

  g_return_if_fail (FOUNDRY_IS_CLI_COMMAND_TREE (self));
  g_return_if_fail (path != NULL);
  g_return_if_fail (path[0] != NULL);
  g_return_if_fail (command != NULL);

  n_parts = g_strv_length ((char **)path);

  data = g_new0 (FoundryCliCommandTreeData, 1);
  data->name = g_strdup (path[n_parts-1]);
  data->command = foundry_cli_command_copy (command);

  insert_node (self->root, path, data);
}

static void
print_recurse (const GNode *node,
               int          depth)
{
  if (node->parent != NULL)
    {
      const FoundryCliCommandTreeData *data = node->data;
      for (int i = 0; i < depth; i++)
        g_print ("  ");
      g_print ("%s\n", data->name);
    }

  for (const GNode *child = node->children; child; child = child->next)
    print_recurse (child, depth + 1);
}

void
_foundry_cli_command_tree_print (FoundryCliCommandTree *self)
{
  g_return_if_fail (FOUNDRY_IS_CLI_COMMAND_TREE (self));

  print_recurse (self->root, -1);
}

static void
clear_entry_data (gpointer data)
{
  GOptionEntry *entry = data;

  switch (entry->arg)
    {
      case G_OPTION_ARG_NONE:
      case G_OPTION_ARG_INT:
      case G_OPTION_ARG_INT64:
      case G_OPTION_ARG_DOUBLE:
        break;

      case G_OPTION_ARG_FILENAME:
      case G_OPTION_ARG_STRING:
        g_clear_pointer ((char **)entry->arg_data, g_free);
        break;

      case G_OPTION_ARG_FILENAME_ARRAY:
      case G_OPTION_ARG_STRING_ARRAY:
        g_clear_pointer ((char ***)entry->arg_data, g_strfreev);
        break;

      case G_OPTION_ARG_CALLBACK:
      default:
        g_assert_not_reached ();
    }

  g_clear_pointer (&entry->arg_data, g_free);
}

static GNode *
lookup_recurse (GNode               *node,
                gboolean             for_completion,
                char              ***args,
                FoundryCliOptions   *options,
                GError             **error)
{
  const char * const *argv = (const char * const *)*args;
  FoundryCliCommandTreeData *data;

  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);

  if (node == NULL)
    return NULL;

  data = node->data;

  if (data->command != NULL && data->command->options != NULL)
    {
      g_autoptr(GOptionContext) context = g_option_context_new (NULL);
      g_autoptr(GArray) entries = g_array_new (TRUE, TRUE, sizeof (GOptionEntry));
      g_autoptr(GError) local_error = NULL;

      g_array_set_clear_func (entries, clear_entry_data);

      g_option_context_set_strict_posix (context, TRUE);
      g_option_context_set_help_enabled (context, FALSE);
      g_option_context_set_ignore_unknown_options (context, for_completion);

      for (const GOptionEntry *entry = data->command->options;
           entry->long_name != NULL;
           entry++)
        {
          GOptionEntry copy = *entry;

          if (entry->arg == G_OPTION_ARG_CALLBACK)
            {
              g_critical ("G_OPTION_ARG_CALLBACK is not supported");
              continue;
            }

          switch (entry->arg)
            {
            case G_OPTION_ARG_NONE:
              copy.arg_data = g_new (gboolean, 1);
              *(gboolean *)copy.arg_data = -1;
              break;

            case G_OPTION_ARG_INT:
              copy.arg_data = g_new (int, 1);
              *(int *)copy.arg_data = 0;
              break;

            case G_OPTION_ARG_INT64:
              copy.arg_data = g_new (gint64, 1);
              *(gint64 *)copy.arg_data = 0;
              break;

            case G_OPTION_ARG_DOUBLE:
              copy.arg_data = g_new (double, 1);
              *(double *)copy.arg_data = .0;
              break;

            case G_OPTION_ARG_FILENAME:
            case G_OPTION_ARG_FILENAME_ARRAY:
            case G_OPTION_ARG_STRING:
            case G_OPTION_ARG_STRING_ARRAY:
              copy.arg_data = g_new0 (char *, 1);
              break;

            case G_OPTION_ARG_CALLBACK:
            default:
              g_assert_not_reached ();
            }

          g_array_append_val (entries, copy);
        }

      if (entries->len > 0)
        g_option_context_add_main_entries (context,
                                           (const GOptionEntry *)(gpointer)entries->data,
                                           data->command->gettext_package ?
                                             data->command->gettext_package :
                                             GETTEXT_PACKAGE);

      if (data->command->prepare != NULL)
        data->command->prepare (context);

      if (!g_option_context_parse_strv (context, args, &local_error))
        {
          if (!for_completion ||
              !g_error_matches (local_error, G_OPTION_ERROR, G_OPTION_ERROR_BAD_VALUE))
            {
              g_propagate_error (error, g_steal_pointer (&local_error));
              return NULL;
            }

          g_clear_error (&local_error);
        }

      argv = (const char * const *)*args;

      g_assert (args != NULL);
      g_assert (args[0] != NULL);

      for (guint i = 0; i < entries->len; i++)
        {
          const GOptionEntry *entry = &g_array_index (entries, GOptionEntry, i);

          switch (entry->arg)
            {
            case G_OPTION_ARG_NONE:
              if (*(gboolean *)entry->arg_data != -1)
                foundry_cli_options_set_boolean (options,
                                                 entry->long_name,
                                                 *(gboolean *)entry->arg_data);
              break;

            case G_OPTION_ARG_INT:
              if (*(int *)entry->arg_data != 0)
                foundry_cli_options_set_int (options,
                                             entry->long_name,
                                             *(int *)entry->arg_data);
              break;

            case G_OPTION_ARG_INT64:
              if (*(gint64 *)entry->arg_data != 0)
                foundry_cli_options_set_int64 (options,
                                               entry->long_name,
                                               *(gint64 *)entry->arg_data);
              break;

            case G_OPTION_ARG_DOUBLE:
              if (*(double *)entry->arg_data != .0)
                foundry_cli_options_set_int64 (options,
                                               entry->long_name,
                                               *(double *)entry->arg_data);
              break;

            case G_OPTION_ARG_FILENAME:
              if (*(const char **)entry->arg_data != NULL)
                foundry_cli_options_set_filename (options,
                                                  entry->long_name,
                                                  *(const char **)entry->arg_data);
              break;

            case G_OPTION_ARG_FILENAME_ARRAY:
              if (*(const char * const **)entry->arg_data != NULL)
                foundry_cli_options_set_filename_array (options,
                                                        entry->long_name,
                                                        *(const char * const **)entry->arg_data);
              break;

            case G_OPTION_ARG_STRING:
              if (*(const char **)entry->arg_data != NULL)
                foundry_cli_options_set_string (options,
                                                entry->long_name,
                                                *(const char **)entry->arg_data);
              break;

            case G_OPTION_ARG_STRING_ARRAY:
              if (*(const char * const **)entry->arg_data != NULL)
                foundry_cli_options_set_string_array (options,
                                                      entry->long_name,
                                                      *(const char * const **)entry->arg_data);
              break;

            case G_OPTION_ARG_CALLBACK:
            default:
              g_assert_not_reached ();
            }
        }
    }

  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);
  g_assert (argv == (const char * const *)*args);

  if (argv[1] != NULL && argv[1][0] != '-')
    {
      for (GNode *child = node->children; child; child = child->next)
        {
          const FoundryCliCommandTreeData *child_data = child->data;

          g_assert (child_data != NULL);
          g_assert (child_data->name != NULL);

          if (g_str_equal (child_data->name, argv[1]))
            {
              g_auto(GStrv) new_args = g_strdupv ((char **)&argv[1]);
              GNode *ret;

              g_free (new_args[0]);
              new_args[0] = g_strdup_printf ("%s-%s", argv[0], argv[1]);

              if ((ret = lookup_recurse (child, for_completion, &new_args, options, error)))
                {
                  g_strfreev (*args);
                  *args = g_steal_pointer (&new_args);
                }

              return ret;
            }
        }
    }

  return node;
}

G_GNUC_WARN_UNUSED_RESULT
static char **
truncate_strv (char  **strv,
               gsize   len)
{
  for (gsize i = len; strv[i]; i++)
    g_clear_pointer (&strv[i], g_free);
  return strv;
}

G_GNUC_WARN_UNUSED_RESULT
static char **
join_strv (char **first,
           char **second)
{
  char **res = g_new0 (char *, g_strv_length (first) + g_strv_length (second) + 1);
  gsize j = 0;

  for (gsize i = 0; first[i]; i++)
    res[j++] = g_steal_pointer (&first[i]);

  for (gsize i = 0; second[i]; i++)
    res[j++] = g_steal_pointer (&second[i]);

  res[j] = NULL;

  g_free (first);
  g_free (second);

  return res;
}


static GNode *
foundry_cli_command_tree_lookup_full (FoundryCliCommandTree   *self,
                                      gboolean                 for_completion,
                                      char                  ***args,
                                      FoundryCliOptions      **options,
                                      GError                 **error)
{
  g_autoptr(FoundryCliOptions) parsed = foundry_cli_options_new ();
  g_auto(GStrv) suffix = NULL;
  GNode *node;

  /* If we come across a "--", strip everything starting from that and then
   * we'll join it at the end. We don't want any internal processing to
   * take that into account.
   */
  for (guint i = 0; (*args)[i]; i++)
    {
      if (g_str_equal ((*args)[i], "--"))
        {
          suffix = g_strdupv (&(*args)[i]);
          *args = truncate_strv (*args, i);
          break;
        }
    }

  if ((node = lookup_recurse (self->root->children, for_completion, args, parsed, error)))
    *options = g_steal_pointer (&parsed);

  if (suffix != NULL)
    *args = join_strv (*args, g_steal_pointer (&suffix));

  return node;
}

const FoundryCliCommand *
foundry_cli_command_tree_lookup (FoundryCliCommandTree   *self,
                                 char                  ***args,
                                 FoundryCliOptions      **options,
                                 GError                 **error)
{
  GNode *node;

  g_return_val_if_fail (FOUNDRY_IS_CLI_COMMAND_TREE (self), NULL);
  g_return_val_if_fail (args != NULL, NULL);
  g_return_val_if_fail (options != NULL, NULL);

  if ((node = foundry_cli_command_tree_lookup_full (self, FALSE, args, options, error)))
    {
      const FoundryCliCommandTreeData *data = node->data;

      if (data->command != NULL)
        return data->command;
    }

  g_set_error (error,
               G_IO_ERROR,
               G_IO_ERROR_NOT_SUPPORTED,
               _("No such command"));

  return NULL;
}

static const GOptionEntry *
find_entry (const GOptionEntry *entries,
            const char         *arg)
{
  if (arg == NULL)
    return NULL;

  if (arg[0] != '-')
    return NULL;

  /* Commonly used as a separator */
  if (g_str_equal (arg, "--"))
    return NULL;

  if (arg[1] == '-')
    {
      for (guint i = 0; entries[i].long_name; i++)
        {
          if (g_str_equal (&arg[2], entries[i].long_name))
            return &entries[i];
        }
    }
  else if (arg[1] != 0)
    {
      const char *next = g_utf8_next_char (arg);
      gunichar ch = g_utf8_get_char (next);

      if (*g_utf8_next_char (next) == 0)
        {
          for (guint i = 0; entries[i].long_name; i++)
            {
              if (ch == entries[i].short_name)
                return &entries[i];
            }
        }
    }

  return NULL;
}

/**
 * foundry_cli_command_tree_complete:
 * @self: a [class@Foundry.CliCommandTree]
 *
 * Returns: (transfer full):
 */
char **
foundry_cli_command_tree_complete (FoundryCliCommandTree *self,
                                   FoundryCommandLine    *command_line,
                                   const char            *line,
                                   int                    point,
                                   const char            *current)
{
  const FoundryCliCommandTreeData *data;
  g_autoptr(FoundryCliOptions) options = NULL;
  g_autoptr(GStrvBuilder) results = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *to_point = NULL;
  g_auto(GStrv) argv = NULL;
  GNode *node = NULL;
  int argc;

  g_return_val_if_fail (FOUNDRY_IS_CLI_COMMAND_TREE (self), NULL);
  g_return_val_if_fail (line != NULL, NULL);

  if (point < 0 || point > strlen (line))
    point = strlen (line);

  to_point = g_strndup (line, point);

  if (!g_shell_parse_argv (to_point, &argc, &argv, NULL))
    return NULL;

  /* ignore_unknown_options will strip things like "--" out, so we have
   * to use @current to potentially complete those.
   */
  if (!(node = foundry_cli_command_tree_lookup_full (self, TRUE, &argv, &options, &error)))
    return NULL;

  g_assert (argv != NULL);
  g_assert (argv[0] != NULL);
  g_assert (options != NULL);
  g_assert (node != NULL);

  argc = g_strv_length (argv);
  data = node->data;
  results = g_strv_builder_new ();

  if (data->command != NULL && data->command->complete)
    {
      g_auto(GStrv) completions = data->command->complete (command_line, argv[0], NULL, options, (const char * const *)argv, current);

      if (completions != NULL)
        g_strv_builder_addv (results, (const char **)completions);
    }

  if (argv[1] == NULL && g_strcmp0 (data->name, current) == 0)
    {
      g_autofree char *with_space = g_strdup_printf ("%s ", data->name);
      g_strv_builder_add (results, with_space);
      return g_strv_builder_end (results);
    }

  for (GNode *child = node->children; child; child = child->next)
    {
      const FoundryCliCommandTreeData *child_data = child->data;

      /* If just starting command name or completing existing command name,
       * then this command is a potential completion.
       */
      if (argv[1] == NULL ||
          (argv[2] == NULL &&
           has_prefix_or_equal (child_data->name, argv[1])))
        {
          g_autofree char *with_space = g_strdup_printf ("%s ", child_data->name);
          g_strv_builder_add (results, with_space);
        }
    }

  if (current != NULL &&
      data->command != NULL &&
      data->command->options != NULL)
    {
      /* Try to complete long commands */
      if (has_prefix_or_equal (current, "-"))
        {
          const char *name = current[1] ? &current[2] : "";

          for (guint i = 0; data->command->options[i].long_name; i++)
            {
              const GOptionEntry *entry = &data->command->options[i];

              if (has_prefix_or_equal (entry->long_name, name))
                {
                  gboolean has_value = entry->arg != G_OPTION_ARG_NONE;
                  g_autofree char *dashed = g_strdup_printf ("--%s%s", entry->long_name, has_value ? "=" : " ");
                  g_strv_builder_add (results, dashed);
                }
            }
        }

      /* Try to complete short commands */
      if (has_prefix_or_equal (current, "-") && current[1] != '-')
        {
          gunichar ch = g_utf8_get_char (g_utf8_next_char (current));

          for (guint i = 0; data->command->options[i].long_name; i++)
            {
              const GOptionEntry *entry = &data->command->options[i];

              if (entry->short_name == 0)
                continue;

              if (!ch || entry->short_name == ch)
                {
                  g_autofree char *dashed = g_strdup_printf ("-%c", entry->short_name);
                  g_strv_builder_add (results, dashed);
                }
            }
        }
    }

  /* If the final argv element is a switch, then find the type of it
   * and see if it takes an argument we know about.
   */
  if (is_null_or_empty (current) &&
      !is_null_or_empty (argv[argc-1]) &&
      argv[argc-1][0] == '-')
    {
      const GOptionEntry *entry;

      if ((entry = find_entry (data->command->options, argv[argc-1])))
        {
          if (entry->arg == G_OPTION_ARG_FILENAME ||
              entry->arg == G_OPTION_ARG_FILENAME_ARRAY)
            {
              g_strv_builder_add (results, "__FOUNDRY_FILE");
            }
          else if (entry->arg == G_OPTION_ARG_STRING ||
                   entry->arg == G_OPTION_ARG_STRING_ARRAY)
            {
              if (data->command->complete)
                {
                  g_auto(GStrv) completions = data->command->complete (command_line, argv[0], entry, options, (const char * const *)argv, current);

                  if (completions != NULL)
                    g_strv_builder_addv (results, (const char **)completions);
                }
            }
        }
    }

  return g_strv_builder_end (results);
}

/**
 * foundry_cli_command_tree_get_default:
 *
 * Gets the default instance for use by the foundry CLI tool.
 *
 * Returns: (transfer none): a #FoundryCliCommandTree
 */
FoundryCliCommandTree *
foundry_cli_command_tree_get_default (void)
{
  static FoundryCliCommandTree *instance;

  if (g_once_init_enter (&instance))
    g_once_init_leave (&instance, foundry_cli_command_tree_new ());

  return instance;
}
