/* foundry-internal-tweak.c
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

#include "foundry-input-combo.h"
#include "foundry-input-choice.h"
#include "foundry-input-font.h"
#include "foundry-input-spin.h"
#include "foundry-input-switch.h"
#include "foundry-internal-tweak.h"
#include "foundry-settings.h"
#include "foundry-string-object-private.h"
#include "foundry-tweak-info.h"
#include "foundry-tweak-info-private.h"
#include "foundry-util.h"

struct _FoundryInternalTweak
{
  FoundryTweak      parent_instance;
  char             *path;
  FoundryTweakInfo *info;
  const char       *gettext_domain;
  GSettings        *settings;
};

G_DEFINE_FINAL_TYPE (FoundryInternalTweak, foundry_internal_tweak, FOUNDRY_TYPE_TWEAK)

static char *
foundry_internal_tweak_dup_title (FoundryTweak *tweak)
{
  FoundryInternalTweak *self = FOUNDRY_INTERNAL_TWEAK (tweak);

  return g_strdup (g_dgettext (self->gettext_domain, self->info->title));
}

static char *
foundry_internal_tweak_dup_subtitle (FoundryTweak *tweak)
{
  FoundryInternalTweak *self = FOUNDRY_INTERNAL_TWEAK (tweak);

  return g_strdup (g_dgettext (self->gettext_domain, self->info->subtitle));
}

static char *
foundry_internal_tweak_dup_display_hint (FoundryTweak *tweak)
{
  FoundryInternalTweak *self = FOUNDRY_INTERNAL_TWEAK (tweak);

  if (self->info->display_hint)
    return g_strdup (self->info->display_hint);

  if (self->info->type == FOUNDRY_TWEAK_TYPE_GROUP)
    return g_strdup ("group");

  return NULL;
}

static char *
foundry_internal_tweak_dup_sort_key (FoundryTweak *tweak)
{
  return g_strdup (FOUNDRY_INTERNAL_TWEAK (tweak)->info->sort_key);
}

static char *
foundry_internal_tweak_dup_path (FoundryTweak *tweak)
{
  return g_strdup (FOUNDRY_INTERNAL_TWEAK (tweak)->path);
}

static char *
foundry_internal_tweak_dup_section (FoundryTweak *tweak)
{
  return g_strdup (FOUNDRY_INTERNAL_TWEAK (tweak)->info->section);
}

static GIcon *
foundry_internal_tweak_dup_icon (FoundryTweak *tweak)
{
  FoundryInternalTweak *self = FOUNDRY_INTERNAL_TWEAK (tweak);

  if (self->info->icon_name)
    {
      GIcon *icon = g_icon_new_for_string (self->info->icon_name, NULL);

      if (icon != NULL)
        return icon;

      return g_themed_icon_new (self->info->icon_name);
    }

  return NULL;
}

static GSettings *
create_settings (FoundryInternalTweak *self,
                 FoundryContext       *context,
                 const char           *schema_id,
                 const char           *path)
{
  g_autoptr(FoundrySettings) settings = NULL;

  g_assert (FOUNDRY_IS_CONTEXT (context));
  g_assert (schema_id != NULL);

  if (path != NULL)
    settings = foundry_settings_new_with_path (context, schema_id, path);
  else
    settings = foundry_settings_new (context, schema_id);

  if (g_str_has_prefix (self->path, "/app/"))
    return foundry_settings_dup_layer (settings, FOUNDRY_SETTINGS_LAYER_APPLICATION);

  if (g_str_has_prefix (self->path, "/project/"))
    return foundry_settings_dup_layer (settings, FOUNDRY_SETTINGS_LAYER_PROJECT);

  if (g_str_has_prefix (self->path, "/user/"))
    return foundry_settings_dup_layer (settings, FOUNDRY_SETTINGS_LAYER_USER);

  g_return_val_if_reached (NULL);
}

static FoundryInput *
create_switch (const FoundryTweakInfo *info,
               GSettings              *settings,
               const char             *key)
{
  FoundryInput *input;
  gboolean value;

  g_assert (info != NULL);
  g_assert (G_IS_SETTINGS (settings));
  g_assert (key != NULL);

  value = g_settings_get_boolean (settings, key);
  input = foundry_input_switch_new (info->title, info->subtitle, NULL, value);

  g_settings_bind (settings, key, input, "value", G_SETTINGS_BIND_DEFAULT);

  return input;
}

static FoundryInput *
create_font (const FoundryTweakInfo *info,
             GSettings              *settings,
             const char             *key)
{
  g_autofree char *value = NULL;
  FoundryInput *input;

  g_assert (info != NULL);
  g_assert (G_IS_SETTINGS (settings));
  g_assert (key != NULL);

  value = g_settings_get_string (settings, key);
  input = foundry_input_font_new (info->title, info->subtitle, NULL, value,
                                  !!(info->flags & FOUNDRY_TWEAK_INFO_FONT_MONOSPACE));

  g_settings_bind (settings, key, input, "value", G_SETTINGS_BIND_DEFAULT);

  return input;
}

static double
get_value_as_double (GVariant *value)
{
  if (g_variant_is_of_type (value, G_VARIANT_TYPE_DOUBLE))
    return g_variant_get_double (value);

  else if (g_variant_is_of_type (value, G_VARIANT_TYPE_INT16))
    return g_variant_get_int16 (value);
  else if (g_variant_is_of_type (value, G_VARIANT_TYPE_UINT16))
    return g_variant_get_uint16 (value);

  else if (g_variant_is_of_type (value, G_VARIANT_TYPE_INT32))
    return g_variant_get_int32 (value);
  else if (g_variant_is_of_type (value, G_VARIANT_TYPE_UINT32))
    return g_variant_get_uint32 (value);

  else if (g_variant_is_of_type (value, G_VARIANT_TYPE_INT64))
    return g_variant_get_int64 (value);
  else if (g_variant_is_of_type (value, G_VARIANT_TYPE_UINT64))
    return g_variant_get_uint64 (value);

  return 0;
}

static FoundryInput *
create_spin (const FoundryTweakInfo *info,
             GSettingsSchemaKey     *schema_key,
             const GVariantType     *type,
             GSettings              *settings,
             const char             *key)
{
  g_autoptr(GVariant) variant = NULL;
  g_autoptr(GVariant) lval = NULL;
  g_autoptr(GVariant) uval = NULL;
  g_autoptr(GVariant) range = NULL;
  g_autoptr(GVariant) values = NULL;
  g_autofree char *range_type = NULL;
  FoundryInput *input;
  double lower = .0;
  double upper = .0;
  double value = .0;
  GVariantIter iter;
  guint n_digits = 0;

  g_assert (info != NULL);
  g_assert (G_IS_SETTINGS (settings));
  g_assert (key != NULL);

  range = g_settings_schema_key_get_range (schema_key);
  g_variant_get (range, "(sv)", &range_type, &values);

  if (!foundry_str_equal0 (range_type, "range") ||
      (2 == g_variant_iter_init (&iter, values)))
    {
      lval = g_variant_iter_next_value (&iter);
      uval = g_variant_iter_next_value (&iter);

      lower = get_value_as_double (lval);
      upper = get_value_as_double (uval);
    }

  if (g_variant_type_equal (type, G_VARIANT_TYPE_DOUBLE))
    n_digits = 2;

  variant = g_settings_get_value (settings, key);
  value = get_value_as_double (variant);

  input = foundry_input_spin_new (info->title, info->subtitle, NULL, value,
                                  lower, upper, n_digits);

  g_settings_bind (settings, key, input, "value", G_SETTINGS_BIND_DEFAULT);

  return input;
}

static void
notify_choice_cb (FoundryInputCombo *combo,
                  GParamSpec        *pspec,
                  GSettings         *settings)
{
  g_autoptr(FoundryInputChoice) choice = NULL;
  const char *key;

  g_assert (FOUNDRY_IS_INPUT_COMBO (combo));
  g_assert (G_IS_SETTINGS (settings));

  if (!(key = g_object_get_data (G_OBJECT (combo), "KEY")))
    return;

  if ((choice = foundry_input_combo_dup_choice (combo)))
    {
      g_autoptr(GObject) item = NULL;

      if ((item = foundry_input_choice_dup_item (choice)))
        {
          if (FOUNDRY_IS_STRING_OBJECT (item))
            {
              const char *string = foundry_string_object_get_string (FOUNDRY_STRING_OBJECT (item));

              g_settings_set_string (settings, key, string);
            }
        }
    }
}

static FoundryInput *
create_combo (const FoundryTweakInfo *info,
              GSettingsSchemaKey     *schema_key,
              const GVariantType     *type,
              GSettings              *settings,
              const char             *key)
{
  g_autoptr(GVariant) range = NULL;
  g_autoptr(GVariant) values = NULL;
  g_autofree char *range_type = NULL;

  g_assert (info != NULL);
  g_assert (G_IS_SETTINGS (settings));
  g_assert (key != NULL);

  range = g_settings_schema_key_get_range (schema_key);
  g_variant_get (range, "(sv)", &range_type, &values);

  if (foundry_str_equal0 (range_type, "enum"))
    {
      g_autoptr(GListStore) choices = g_list_store_new (G_TYPE_OBJECT);
      g_autoptr(FoundryInput) default_choice = NULL;
      g_autofree char *current = g_settings_get_string (settings, key);
      FoundryInput *ret = NULL;
      GVariantIter iter;

      /* This sucks because we can't currently translate it. But having
       * everything create their own combos automatically is a _lot_ of
       * code I don't want to write.
       *
       * So instead, if you find yourself here, help come up with ideas
       * on how we can avoid that.
       */

      if (g_variant_iter_init (&iter, values))
        {
          char *word = NULL;

          while (g_variant_iter_loop (&iter, "s", &word))
            {
              g_autoptr(FoundryStringObject) item = foundry_string_object_new (word);
              g_autoptr(FoundryInput) choice = foundry_input_choice_new (word, NULL, G_OBJECT (item));

              if (foundry_str_equal0 (word, current))
                default_choice = g_object_ref (choice);

              g_list_store_append (choices, choice);
            }
        }

      ret = foundry_input_combo_new (info->title,
                                     info->subtitle,
                                     NULL,
                                     G_LIST_MODEL (choices));

      foundry_input_combo_set_choice (FOUNDRY_INPUT_COMBO (ret),
                                      FOUNDRY_INPUT_CHOICE (default_choice));

      g_object_set_data_full (G_OBJECT (ret), "KEY", g_strdup (key), g_free);

      g_signal_connect_object (ret,
                               "notify::choice",
                               G_CALLBACK (notify_choice_cb),
                               settings,
                               0);

      return ret;
    }

  return NULL;
}

static FoundryInput *
foundry_internal_tweak_create_input (FoundryTweak   *tweak,
                                     FoundryContext *context)
{
  FoundryInternalTweak *self = FOUNDRY_INTERNAL_TWEAK (tweak);

  g_assert (FOUNDRY_IS_INTERNAL_TWEAK (self));
  g_assert (FOUNDRY_IS_CONTEXT (context));
  g_assert (self->info != NULL);

  if (self->info->source == NULL)
    return NULL;

  if (self->info->source->type == FOUNDRY_TWEAK_SOURCE_TYPE_CALLBACK)
    {
      g_autofree char *path = foundry_tweak_dup_path (tweak);
      return self->info->source->callback.callback (self->info, path, context);
    }

  if (self->info->source->type == FOUNDRY_TWEAK_SOURCE_TYPE_SETTING)
    {
      g_autoptr(GSettingsSchemaKey) key = NULL;
      g_autoptr(GSettingsSchema) schema = NULL;
      const GVariantType *value_type;
      const char *key_name;

      if (self->settings == NULL)
        self->settings = create_settings (self,
                                          context,
                                          self->info->source->setting.schema_id,
                                          self->info->source->setting.path);

      if (self->settings == NULL)
        return NULL;

      g_object_get (self->settings,
                    "settings-schema", &schema,
                    NULL);

      key_name = self->info->source->setting.key;

      if (!(key = g_settings_schema_get_key (schema, key_name)))
        return NULL;

      value_type = g_settings_schema_key_get_value_type (key);

      if (self->info->type == FOUNDRY_TWEAK_TYPE_SWITCH &&
          g_variant_type_equal (value_type, G_VARIANT_TYPE_BOOLEAN))
        return create_switch (self->info, self->settings, key_name);
      else if (self->info->type == FOUNDRY_TWEAK_TYPE_FONT &&
               g_variant_type_equal (value_type, G_VARIANT_TYPE_STRING))
        return create_font (self->info, self->settings, key_name);
      else if (self->info->type == FOUNDRY_TWEAK_TYPE_SPIN)
        return create_spin (self->info, key, value_type, self->settings, key_name);
      else if (self->info->type == FOUNDRY_TWEAK_TYPE_COMBO)
        return create_combo (self->info, key, value_type, self->settings, key_name);
    }

  return NULL;
}

static void
foundry_internal_tweak_finalize (GObject *object)
{
  FoundryInternalTweak *self = (FoundryInternalTweak *)object;

  g_clear_pointer (&self->info, foundry_tweak_info_unref);
  g_clear_pointer (&self->path, g_free);

  G_OBJECT_CLASS (foundry_internal_tweak_parent_class)->finalize (object);
}

static void
foundry_internal_tweak_class_init (FoundryInternalTweakClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryTweakClass *tweak_class = FOUNDRY_TWEAK_CLASS (klass);

  object_class->finalize = foundry_internal_tweak_finalize;

  tweak_class->dup_path = foundry_internal_tweak_dup_path;
  tweak_class->dup_title = foundry_internal_tweak_dup_title;
  tweak_class->dup_subtitle = foundry_internal_tweak_dup_subtitle;
  tweak_class->dup_display_hint = foundry_internal_tweak_dup_display_hint;
  tweak_class->dup_sort_key = foundry_internal_tweak_dup_sort_key;
  tweak_class->dup_section = foundry_internal_tweak_dup_section;
  tweak_class->dup_icon = foundry_internal_tweak_dup_icon;
  tweak_class->create_input = foundry_internal_tweak_create_input;
}

static void
foundry_internal_tweak_init (FoundryInternalTweak *self)
{
}

FoundryTweak *
foundry_internal_tweak_new (const char       *gettext_domain,
                            FoundryTweakInfo *info,
                            char             *path)
{
  FoundryInternalTweak *self;

  g_return_val_if_fail (info != NULL, NULL);
  g_return_val_if_fail (path != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_INTERNAL_TWEAK, NULL);
  self->gettext_domain = gettext_domain ? g_intern_string (gettext_domain) : GETTEXT_PACKAGE;
  self->info = info;
  self->path = path;

  return FOUNDRY_TWEAK (self);
}
