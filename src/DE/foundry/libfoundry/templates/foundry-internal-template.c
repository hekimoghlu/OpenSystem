/* foundry-internal-template.c
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

#include <tmpl-glib.h>

#include "foundry-context.h"
#include "foundry-input-choice.h"
#include "foundry-input-combo.h"
#include "foundry-input-file.h"
#include "foundry-input-group.h"
#include "foundry-input-switch.h"
#include "foundry-input-text.h"
#include "foundry-input-validator-regex.h"
#include "foundry-language.h"
#include "foundry-license.h"
#include "foundry-internal-template-private.h"
#include "foundry-string-object-private.h"
#include "foundry-template-locator-private.h"
#include "foundry-template-output.h"
#include "foundry-template-util-private.h"
#include "foundry-util.h"

#include "line-reader-private.h"

struct _FoundryInternalTemplate
{
  FoundryTemplate parent_instance;
  FoundryContext *context;
  FoundryInput *input;
  FoundryInput *location;
  char **tags;
  char *description;
  char *id;
  GArray *files;
  guint is_project : 1;
};

G_DEFINE_FINAL_TYPE (FoundryInternalTemplate, foundry_internal_template, FOUNDRY_TYPE_TEMPLATE)

#define has_prefix(str,len,prefix) \
  ((len >= strlen(prefix)) && (memcmp(str,prefix,strlen(prefix)) == 0))

typedef struct _File
{
  char   **conditions;
  char    *pattern;
  GBytes  *bytes;
} File;

static void
file_clear (File *file)
{
  g_clear_pointer (&file->pattern, g_free);
  g_clear_pointer (&file->bytes, g_bytes_unref);
  g_clear_pointer (&file->conditions, g_strfreev);
}

static char **
copy_conditions (GPtrArray *ar)
{
  char **ret;

  if (ar == NULL || ar->len == 0)
    return NULL;

  ret = g_new0 (char *, ar->len + 1);

  for (guint i = 0; i < ar->len; i++)
    ret[i] = g_strdup (g_ptr_array_index (ar, i));
  ret[ar->len] = NULL;

  return ret;
}

static char *
foundry_internal_template_dup_id (FoundryTemplate *template)
{
  return g_strdup (FOUNDRY_INTERNAL_TEMPLATE (template)->id);
}

static char *
foundry_internal_template_dup_description (FoundryTemplate *template)
{
  return g_strdup (FOUNDRY_INTERNAL_TEMPLATE (template)->description);
}

static FoundryInput *
foundry_internal_template_dup_input (FoundryTemplate *template)
{
  FoundryInternalTemplate *self = FOUNDRY_INTERNAL_TEMPLATE (template);

  return self->input ? g_object_ref (self->input) : NULL;
}

static char **
foundry_internal_template_dup_tags (FoundryTemplate *template)
{
  return g_strdupv (FOUNDRY_INTERNAL_TEMPLATE (template)->tags);
}

static void
add_input_to_scope (TmplScope              *scope,
                    FoundryInput           *input,
                    FoundryTemplateLocator *locator)
{
  const char *name = g_object_get_data (G_OBJECT (input), "VARIABLE");
  GObjectClass *klass = G_OBJECT_GET_CLASS (input);
  g_auto(GValue) value = G_VALUE_INIT;
  GParamSpec *pspec;

  if (FOUNDRY_IS_INPUT_GROUP (input))
    {
      g_autoptr(GListModel) children = foundry_input_group_list_children (FOUNDRY_INPUT_GROUP (input));
      guint n_items = g_list_model_get_n_items (children);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryInput) child = g_list_model_get_item (children, i);

          add_input_to_scope (scope, child, locator);
        }

      return;
    }

  if (FOUNDRY_IS_INPUT_COMBO (input))
    {
      g_autoptr(FoundryInputChoice) choice = NULL;
      g_autoptr(GObject) item = NULL;
      g_autoptr(GBytes) license_bytes = NULL;

      if ((choice = foundry_input_combo_dup_choice (FOUNDRY_INPUT_COMBO (input))) &&
          (item = foundry_input_choice_dup_item (choice)) &&
          FOUNDRY_IS_LICENSE (item) &&
          (license_bytes = foundry_license_dup_snippet_text (FOUNDRY_LICENSE (item))))
        foundry_template_locator_set_license_text (locator, license_bytes);
    }

  if ((pspec = g_object_class_find_property (klass, "value")))
    {
      g_value_init (&value, pspec->value_type);
      g_object_get_property (G_OBJECT (input), "value", &value);
    }
  else if ((pspec = g_object_class_find_property (klass, "choice")))
    {
      g_value_init (&value, pspec->value_type);
      g_object_get_property (G_OBJECT (input), "choice", &value);

      if (G_VALUE_HOLDS (&value, FOUNDRY_TYPE_INPUT_CHOICE))
        {
          g_autoptr(FoundryInputChoice) choice = g_value_dup_object (&value);

          g_value_unset (&value);
          g_value_init (&value, G_TYPE_OBJECT);
          g_value_take_object (&value, foundry_input_choice_dup_item (choice));
        }
    }

  if (G_IS_VALUE (&value) && name)
    tmpl_scope_set_value (scope, name, &value);
}

static gboolean
evals_as_true (GValue *value)
{
  if (!G_VALUE_HOLDS_BOOLEAN (value))
    {
      g_auto(GValue) trans = G_VALUE_INIT;

      if (!g_value_type_transformable (G_VALUE_TYPE (value), G_TYPE_BOOLEAN))
        return FALSE;

      g_value_init (&trans, G_TYPE_BOOLEAN);
      if (!g_value_transform (value, &trans))
        return FALSE;

      return g_value_get_boolean (&trans);
    }

  return g_value_get_boolean (value);
}

static DexFuture *
foundry_internal_template_expand_fiber (FoundryInternalTemplate *self,
                                        FoundryLicense          *license)
{
  g_autoptr(TmplTemplateLocator) locator = NULL;
  g_autoptr(TmplScope) parent_scope = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GFile) location = NULL;

  g_assert (FOUNDRY_IS_INTERNAL_TEMPLATE (self));
  g_assert (!license || FOUNDRY_IS_LICENSE (license));

  store = g_list_store_new (FOUNDRY_TYPE_TEMPLATE_OUTPUT);
  location = foundry_input_file_dup_value (FOUNDRY_INPUT_FILE (self->location));
  locator = foundry_template_locator_new ();

  parent_scope = tmpl_scope_new ();
  add_simple_scope (parent_scope);

  if (license != NULL)
    {
      g_autoptr(GBytes) license_bytes = foundry_license_dup_snippet_text (license);

      foundry_template_locator_set_license_text (FOUNDRY_TEMPLATE_LOCATOR (locator), license_bytes);
    }

  {
    g_autoptr(GListModel) children = NULL;
    guint n_items;

    children = foundry_input_group_list_children (FOUNDRY_INPUT_GROUP (self->input));
    n_items = g_list_model_get_n_items (children);

    for (guint i = 0; i < n_items; i++)
      {
        g_autoptr(FoundryInput) child = g_list_model_get_item (children, i);

        add_input_to_scope (parent_scope, child, FOUNDRY_TEMPLATE_LOCATOR (locator));
      }
  }

  for (guint f = 0; f < self->files->len; f++)
    {
      const File *file_info = &g_array_index (self->files, File, f);
      g_autoptr(FoundryTemplateOutput) output = NULL;
      g_autoptr(TmplScope) scope = tmpl_scope_new_with_parent (parent_scope);
      g_autoptr(TmplTemplate) template = NULL;
      g_autoptr(GFile) dest_file = NULL;
      g_autoptr(GError) error = NULL;
      g_autoptr(GBytes) bytes = NULL;
      g_autofree char *expand = NULL;
      g_autofree char *pattern = g_strdup (file_info->pattern);
      g_auto(GValue) return_value = G_VALUE_INIT;
      GBytes *input_bytes = file_info->bytes;

      if (file_info->conditions != NULL)
        {
          gboolean allowed = TRUE;

          for (guint c = 0; file_info->conditions[c]; c++)
            {
              g_autoptr(TmplExpr) expr = tmpl_expr_from_string (file_info->conditions[c], NULL);
              g_auto(GValue) eval = G_VALUE_INIT;

              if (expr == NULL ||
                  !tmpl_expr_eval (expr, parent_scope, &eval, NULL) ||
                  !evals_as_true (&eval))
                {
                  allowed = FALSE;
                  break;
                }
            }

          if (!allowed)
            continue;
        }

      if (strstr (pattern, "{{"))
        {
          g_autoptr(TmplTemplate) expander = tmpl_template_new (NULL);
          g_autofree char *dest_eval = NULL;

          if (!tmpl_template_parse_string (expander, pattern, &error))
            return dex_future_new_for_error (g_steal_pointer (&error));

          if (!(dest_eval = tmpl_template_expand_string (expander, scope, &error)))
            return dex_future_new_for_error (g_steal_pointer (&error));

          g_set_str (&pattern, dest_eval);
        }

      dest_file = g_file_get_child (location, pattern);

      /* This might be a directory creation */
      if (g_str_has_suffix (pattern, "/") && input_bytes == NULL)
        {
          output = foundry_template_output_new_directory (dest_file);
          g_list_store_append (store, output);
          continue;
        }

      if (foundry_str_empty0 (pattern))
        {
          g_autoptr(TmplExpr) expr = NULL;

          if (!(expr = expr_new_from_bytes (input_bytes, &error)))
            return dex_future_new_for_error (g_steal_pointer (&error));

          if (!tmpl_expr_eval (expr, parent_scope, &return_value, &error))
            return dex_future_new_for_error (g_steal_pointer (&error));

          continue;
        }

      scope_take_string (scope, "filename", g_path_get_basename (pattern));

      template = tmpl_template_new (locator);

      if (!template_parse_bytes (template, input_bytes, &error))
        return dex_future_new_for_error (g_steal_pointer (&error));

      if (!(expand = tmpl_template_expand_string (template, scope, &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      if (!g_file_has_prefix (dest_file, location))
        return dex_future_new_reject (G_IO_ERROR,
                                      G_IO_ERROR_INVAL,
                                      "Cannot create file above location");


      bytes = g_bytes_new_take (expand, strlen (expand)), expand = NULL;
      output = foundry_template_output_new (dest_file, bytes, -1);

      g_list_store_append (store, output);
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
foundry_internal_template_expand (FoundryTemplate *template)
{
  FoundryInternalTemplate *self = (FoundryInternalTemplate *)template;
  g_autoptr(FoundryLicense) default_license = NULL;

  g_assert (FOUNDRY_IS_INTERNAL_TEMPLATE (template));

  if (self->context != NULL)
    default_license = foundry_context_dup_default_license (self->context);

  return foundry_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                  G_CALLBACK (foundry_internal_template_expand_fiber),
                                  2,
                                  FOUNDRY_TYPE_INTERNAL_TEMPLATE, template,
                                  FOUNDRY_TYPE_LICENSE, default_license);
}

static void
foundry_internal_template_finalize (GObject *object)
{
  FoundryInternalTemplate *self = (FoundryInternalTemplate *)object;

  g_clear_object (&self->input);
  g_clear_object (&self->location);
  g_clear_pointer (&self->id, g_free);
  g_clear_pointer (&self->description, g_free);
  g_clear_pointer (&self->files, g_array_unref);
  g_clear_pointer (&self->tags, g_strfreev);

  G_OBJECT_CLASS (foundry_internal_template_parent_class)->finalize (object);
}

static void
foundry_internal_template_class_init (FoundryInternalTemplateClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryTemplateClass *template_class = FOUNDRY_TEMPLATE_CLASS (klass);
  g_autoptr(GError) error = NULL;

  object_class->finalize = foundry_internal_template_finalize;

  template_class->dup_id = foundry_internal_template_dup_id;
  template_class->dup_description = foundry_internal_template_dup_description;
  template_class->dup_input = foundry_internal_template_dup_input;
  template_class->dup_tags = foundry_internal_template_dup_tags;
  template_class->expand = foundry_internal_template_expand;
}

static void
foundry_internal_template_init (FoundryInternalTemplate *self)
{
  self->files = g_array_new (FALSE, FALSE, sizeof (File));
  g_array_set_clear_func (self->files, (GDestroyNotify)file_clear);
}

static FoundryInput *
create_license_combo (const char *default_license)
{
  g_autoptr(GListStore) choices = g_list_store_new (G_TYPE_OBJECT);
  g_autoptr(GListModel) licenses = foundry_license_list_all ();
  g_autoptr(FoundryInput) value = NULL;
  g_autoptr(FoundryInput) ret = NULL;
  guint n_licenses = g_list_model_get_n_items (licenses);

  for (guint i = 0; i < n_licenses; i++)
    {
      g_autoptr(FoundryLicense) license = g_list_model_get_item (licenses, i);
      g_autofree char *id = foundry_license_dup_id (license);
      g_autofree char *title = foundry_license_dup_title (license);
      g_autoptr(FoundryInput) choice = foundry_input_choice_new (title, NULL, G_OBJECT (license));

      if (g_strcmp0 (id, default_license) == 0)
        g_set_object (&value, choice);

      g_list_store_append (choices, choice);
    }

  ret = foundry_input_combo_new (_("License"), NULL, NULL, G_LIST_MODEL (choices));

  if (value != NULL)
    foundry_input_combo_set_choice (FOUNDRY_INPUT_COMBO (ret), FOUNDRY_INPUT_CHOICE (value));

  return g_steal_pointer (&ret);
}

static FoundryInput *
create_language_combo (GKeyFile   *keyfile,
                       const char *group)
{
  g_autofree char *title = g_key_file_get_string (keyfile, group, "Title", NULL);
  g_autofree char *subtitle = g_key_file_get_string (keyfile, group, "Subtitle", NULL);
  g_autofree char *default_str = g_key_file_get_string (keyfile, group, "Default", NULL);
  g_autoptr(GListStore) choices = g_list_store_new (G_TYPE_OBJECT);
  g_auto(GStrv) str_choices = g_key_file_get_string_list (keyfile, group, "Choices", NULL, NULL);
  g_autoptr(FoundryInput) default_choice = NULL;
  FoundryInput *ret;

  if (str_choices == NULL || str_choices[0] == NULL)
    return NULL;

  for (guint i = 0; str_choices[i]; i++)
    {
      g_autoptr(FoundryLanguage) language = NULL;

      if ((language = foundry_language_find (str_choices[i])))
        {
          const char *name = foundry_language_get_name (language);
          g_autoptr(FoundryInput) choice = NULL;

          choice = foundry_input_choice_new (name, NULL, G_OBJECT (language));

          g_list_store_append (choices, choice);

          if (g_strcmp0 (default_str, str_choices[i]) == 0)
            g_set_object (&default_choice, choice);
        }
    }

  if (g_list_model_get_n_items (G_LIST_MODEL (choices)) == 0)
    return NULL;

  ret = foundry_input_combo_new (title, subtitle, NULL, G_LIST_MODEL (choices));

  if (default_choice)
    foundry_input_combo_set_choice (FOUNDRY_INPUT_COMBO (ret),
                                    FOUNDRY_INPUT_CHOICE (default_choice));

  return ret;
}

static FoundryInput *
create_combo (GKeyFile   *keyfile,
              const char *group)
{
  g_autoptr(GListStore) choices = g_list_store_new (G_TYPE_OBJECT);
  g_auto(GStrv) str_choices = g_key_file_get_string_list (keyfile, group, "Choices", NULL, NULL);
  g_autofree char *title = g_key_file_get_string (keyfile, group, "Title", NULL);
  g_autofree char *subtitle = g_key_file_get_string (keyfile, group, "Subtitle", NULL);
  g_autofree char *default_str = g_key_file_get_string (keyfile, group, "Default", NULL);
  g_autoptr(FoundryInput) default_choice = NULL;
  FoundryInput *ret;

  if (str_choices == NULL || str_choices[0] == NULL)
    return NULL;

  for (guint i = 0; str_choices[i]; i++)
    {
      g_autoptr(FoundryStringObject) value = foundry_string_object_new (str_choices[i]);
      g_autoptr(FoundryInput) choice = foundry_input_choice_new (str_choices[i], NULL, G_OBJECT (value));

      g_list_store_append (choices, choice);

      if (g_strcmp0 (str_choices[i], default_str) == 0)
        g_set_object (&default_choice, choice);
    }

  ret = foundry_input_combo_new (title, subtitle, NULL, G_LIST_MODEL (choices));

  if (default_choice)
    foundry_input_combo_set_choice (FOUNDRY_INPUT_COMBO (ret),
                                    FOUNDRY_INPUT_CHOICE (default_choice));

  return ret;
}

static void
create_inputs_from_keyfile (GPtrArray *inputs,
                            GKeyFile  *keyfile)
{
  g_autoptr(GHashTable) input_groups = NULL;
  g_auto(GStrv) groups = NULL;

  g_assert (inputs != NULL);
  g_assert (keyfile != NULL);

  groups = g_key_file_get_groups (keyfile, NULL);
  input_groups = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, (GDestroyNotify)g_ptr_array_unref);

  for (guint i = 0; groups[i]; i++)
    {
      const char *group = groups[i];

      if (g_str_has_prefix (groups[i], "Group"))
        {
          g_hash_table_insert (input_groups,
                               g_strstrip (g_strdup (group + strlen("Group"))),
                               g_ptr_array_new ());
        }
      else if (g_str_has_prefix (groups[i], "Input"))
        {
          g_autofree char *id = g_strstrip (g_strdup (group + strlen ("Input")));
          g_autofree char *type = g_key_file_get_string (keyfile, group, "Type", NULL);
          g_autofree char *title = g_key_file_get_string (keyfile, group, "Title", NULL);
          g_autofree char *subtitle = g_key_file_get_string (keyfile, group, "Subtitle", NULL);
          g_autofree char *value = g_key_file_get_string (keyfile, group, "Default", NULL);
          g_autofree char *input_group_id = g_key_file_get_string (keyfile, group, "Group", NULL);
          g_autoptr(FoundryInput) input = NULL;

          if (strcasecmp (type, "text") == 0)
            {
              g_autofree char *regex_str = g_key_file_get_string (keyfile, group, "Validate", NULL);
              g_autoptr(FoundryInputValidator) validator = NULL;

              if (regex_str)
                {
                  g_autoptr(GError) regex_error = NULL;
                  g_autoptr(GRegex) regex = g_regex_new (regex_str, 0, 0, &regex_error);

                  if (regex_error != NULL)
                    g_warning ("%s: `%s`", regex_error->message, regex_str);
                  else if (regex != NULL)
                    validator = foundry_input_validator_regex_new (regex);
                }

              input = foundry_input_text_new (title, subtitle, g_steal_pointer (&validator), value);
            }
          else if (strcasecmp (type, "switch") == 0)
            {
              gboolean initial_value = FALSE;

              initial_value = value && (*value == 't' || *value == 'T');
              input = foundry_input_switch_new (title, subtitle, NULL, initial_value);
            }
          else if (strcasecmp (type, "license") == 0)
            {
              g_autofree char *default_license = g_key_file_get_string (keyfile, group, "Default", NULL);

              input = create_license_combo (default_license);
            }
          else if (strcasecmp (type, "language") == 0)
            {
              input = create_language_combo (keyfile, group);
            }
          else if (strcasecmp (type, "combo") == 0)
            {
              input = create_combo (keyfile, group);
            }

          if (input != NULL)
            {
              GPtrArray *ar;

              g_object_set_data_full (G_OBJECT (input), "VARIABLE", g_steal_pointer (&id), g_free);

              if (input_group_id &&
                  (ar = g_hash_table_lookup (input_groups, input_group_id)))
                g_ptr_array_add (ar, g_steal_pointer (&input));
              else
                g_ptr_array_add (inputs, g_steal_pointer (&input));
            }
        }
    }

  for (guint i = 0; groups[i]; i++)
    {
      const char *group = groups[i];

      if (g_str_has_prefix (groups[i], "Group"))
        {
          g_autofree char *id = g_strstrip (g_strdup (group + strlen("Group")));
          g_autofree char *title = g_key_file_get_string (keyfile, group, "Title", NULL);
          g_autofree char *subtitle = g_key_file_get_string (keyfile, group, "Subtitle", NULL);
          GPtrArray *ar = g_hash_table_lookup (input_groups, id);

          if (ar->len > 0)
            g_ptr_array_add (inputs,
                             foundry_input_group_new (title, subtitle, NULL,
                                                      (FoundryInput **)ar->pdata,
                                                      ar->len));
        }
    }
}

static DexFuture *
foundry_internal_template_new_fiber (FoundryContext *context,
                                     GFile          *file)
{
  g_autoptr(FoundryInternalTemplate) self = NULL;
  g_autoptr(GPtrArray) inputs = NULL;
  g_autoptr(GPtrArray) conditions = NULL;
  g_autoptr(GKeyFile) keyfile = NULL;
  g_autoptr(GBytes) bytes = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) location = NULL;
  g_autofree char *basename = NULL;
  g_autofree char *title = NULL;
  g_autofree char *subtitle = NULL;
  LineReader reader;
  char *line;
  gsize len;

  g_assert (!context || FOUNDRY_IS_CONTEXT (context));
  g_assert (G_IS_FILE (file));

  conditions = g_ptr_array_new_with_free_func (g_free);
  inputs = g_ptr_array_new_with_free_func (g_object_unref);
  basename = g_file_get_basename (file);
  self = g_object_new (FOUNDRY_TYPE_INTERNAL_TEMPLATE,
                       NULL);

  g_set_object (&self->context, context);

  if (!(bytes = dex_await_boxed (dex_file_load_contents_bytes (file), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (context != NULL)
    location = foundry_context_dup_project_directory (context);
  else
    location = g_file_new_for_path (g_get_current_dir ());

  self->location = foundry_input_file_new (_("Location"), NULL, NULL,
                                           G_FILE_TYPE_DIRECTORY,
                                           location);
  g_ptr_array_add (inputs, g_object_ref (self->location));

  /* First load keyfile from the head of the template */
  line_reader_init_from_bytes (&reader, bytes);
  while ((line = line_reader_next (&reader, &len)))
    {
      if (len &&
          (has_prefix (line, len, "```") | has_prefix (line, len, "if ")))
        {
          g_autoptr(GKeyFile) parsed = g_key_file_new ();
          g_autoptr(GBytes) suffix = NULL;
          const char *begin_keyfile = NULL;
          gsize size;
          gsize offset;

          begin_keyfile = (const char *)g_bytes_get_data (bytes, &size);
          offset = line - begin_keyfile;

          if (!g_key_file_load_from_data (parsed, begin_keyfile, offset, 0, &error))
            return dex_future_new_for_error (g_steal_pointer (&error));

          keyfile = g_steal_pointer (&parsed);
          suffix = g_bytes_new_from_bytes (bytes, offset, size - offset);

          g_clear_pointer (&bytes, g_bytes_unref);
          bytes = g_steal_pointer (&suffix);

          break;
        }
    }

  if (!keyfile)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "Failed to parse keyfile from template");

  line_reader_init_from_bytes (&reader, bytes);

  while ((line = line_reader_next (&reader, &len)))
    {
      if (has_prefix (line, len, "if "))
        {
          g_ptr_array_add (conditions, g_strstrip (g_strndup (line + 3, len - 3)));
        }
      else if (has_prefix (line, len, "end"))
        {
          if (conditions->len > 0)
            g_ptr_array_remove_index (conditions, conditions->len - 1);
        }
      else if (has_prefix (line, len, "```"))
        {
          g_autofree char *file_pattern = g_strndup (line + 3, len - 3);
          g_autoptr(GString) str = g_string_new (NULL);
          File f;

          while ((line = line_reader_next (&reader, &len)))
            {
              if (has_prefix (line, len, "```"))
                break;

              g_string_append_len (str, line, len);
              g_string_append_c (str, '\n');
            }

          f.pattern = g_steal_pointer (&file_pattern);
          f.bytes = g_string_free_to_bytes (g_steal_pointer (&str));
          f.conditions = copy_conditions (conditions);

          g_array_append_val (self->files, f);
        }
      else if (len > 0 && line[len-1] == '/')
        {
          File f;

          f.pattern = g_strndup (line, len);
          f.conditions = copy_conditions (conditions);
          f.bytes = NULL;

          g_array_append_val (self->files, f);
        }
    }

  create_inputs_from_keyfile (inputs, keyfile);

  title = g_key_file_get_string (keyfile, "Template", "Title", NULL);
  subtitle = g_key_file_get_string (keyfile, "Template", "Subtitle", NULL);

  self->input = foundry_input_group_new (title, subtitle, NULL,
                                         (FoundryInput **)(gpointer)inputs->pdata,
                                         inputs->len);

  if (!(self->id = g_key_file_get_string (keyfile, "Template", "Name", NULL)))
    {
      if (g_str_has_suffix (basename, ".template"))
        self->id = g_strndup (basename, strlen (basename) - strlen (".template"));
      if (g_str_has_suffix (basename, ".project"))
        self->id = g_strndup (basename, strlen (basename) - strlen (".project"));
      else
        self->id = g_strdup (basename);
    }

  self->description = g_key_file_get_string (keyfile, "Template", "Description", NULL);
  self->tags = g_key_file_get_string_list (keyfile, "Template", "Tags", NULL, NULL);

  if (g_str_has_suffix (basename, ".project"))
    self->tags = foundry_strv_append (self->tags, "project");
  else if (g_str_has_suffix (basename, ".template"))
    self->tags = foundry_strv_append (self->tags, "code");

  return dex_future_new_take_object (g_steal_pointer (&self));
}

DexFuture *
foundry_internal_template_new (FoundryContext *context,
                               GFile          *file)
{
  dex_return_error_if_fail (!context || FOUNDRY_IS_CONTEXT (context));
  dex_return_error_if_fail (G_IS_FILE (file));

  return foundry_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                  G_CALLBACK (foundry_internal_template_new_fiber),
                                  2,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  G_TYPE_FILE, file);
}
