/* foundry-text-settings.c
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

#include <libpeas.h>

#include "foundry-extension-set.h"
#include "foundry-text-document.h"
#include "foundry-text-settings-private.h"
#include "foundry-text-settings-provider-private.h"
#include "foundry-util.h"

struct _FoundryTextSettings
{
  FoundryContextual parent_instance;

  FoundryExtensionSet *addins;
  char *custom_font;

  GWeakRef document_wr;

  guint completion_page_size;
  guint right_margin_position;
  guint tab_width;
  guint indent_width;
  double line_height;

  guint auto_indent : 1;
  guint completion_auto_select : 1;
  guint enable_completion : 1;
  guint enable_snippets : 1;
  guint enable_spell_check : 1;
  guint highlight_current_line : 1;
  guint highlight_matching_brackets : 1;
  guint implicit_trailing_newline : 1;
  guint indent_on_tab : 1;
  guint insert_matching_brace : 1;
  guint insert_spaces_instead_of_tabs : 1;
  guint override_indent_width : 1;
  guint overwrite_matching_brace : 1;
  guint show_diagnostics : 1;
  guint show_line_changes : 1;
  guint show_line_changes_overview : 1;
  guint show_line_numbers : 1;
  guint show_right_margin : 1;
  guint smart_backspace : 1;
  guint smart_home_end : 1;
  guint use_custom_font : 1;
  guint wrap : 2;

  guint auto_indent_set : 1;
  guint completion_auto_select_set : 1;
  guint completion_page_size_set : 1;
  guint custom_font_set : 1;
  guint enable_completion_set : 1;
  guint enable_snippets_set : 1;
  guint enable_spell_check_set : 1;
  guint highlight_current_line_set : 1;
  guint highlight_matching_brackets_set : 1;
  guint implicit_trailing_newline_set : 1;
  guint indent_on_tab_set : 1;
  guint indent_width_set : 1;
  guint insert_matching_brace_set : 1;
  guint insert_spaces_instead_of_tabs_set : 1;
  guint line_height_set : 1;
  guint override_indent_width_set : 1;
  guint overwrite_matching_brace_set : 1;
  guint right_margin_position_set : 1;
  guint show_diagnostics_set : 1;
  guint show_line_changes_overview_set : 1;
  guint show_line_changes_set : 1;
  guint show_line_numbers_set : 1;
  guint show_right_margin_set : 1;
  guint smart_backspace_set : 1;
  guint smart_home_end_set : 1;
  guint tab_width_set : 1;
  guint use_custom_font_set : 1;
  guint wrap_set : 1;
};

G_DEFINE_FINAL_TYPE (FoundryTextSettings, foundry_text_settings, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_AUTO_INDENT,
  PROP_COMPLETION_AUTO_SELECT,
  PROP_COMPLETION_PAGE_SIZE,
  PROP_CUSTOM_FONT,
  PROP_DOCUMENT,
  PROP_ENABLE_COMPLETION,
  PROP_ENABLE_SNIPPETS,
  PROP_ENABLE_SPELL_CHECK,
  PROP_HIGHLIGHT_CURRENT_LINE,
  PROP_HIGHLIGHT_MATCHING_BRACKETS,
  PROP_IMPLICIT_TRAILING_NEWLINE,
  PROP_INDENT_ON_TAB,
  PROP_INDENT_WIDTH,
  PROP_INSERT_MATCHING_BRACE,
  PROP_INSERT_SPACES_INSTEAD_OF_TABS,
  PROP_LINE_HEIGHT,
  PROP_OVERRIDE_INDENT_WIDTH,
  PROP_OVERWRITE_MATCHING_BRACE,
  PROP_RIGHT_MARGIN_POSITION,
  PROP_SHOW_DIAGNOSTICS,
  PROP_SHOW_LINE_CHANGES,
  PROP_SHOW_LINE_CHANGES_OVERVIEW,
  PROP_SHOW_LINE_NUMBERS,
  PROP_SHOW_RIGHT_MARGIN,
  PROP_SMART_BACKSPACE,
  PROP_SMART_HOME_END,
  PROP_TAB_WIDTH,
  PROP_USE_CUSTOM_FONT,
  PROP_WRAP,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
collect_by_priority_cb (FoundryExtensionSet *set,
                        PeasPluginInfo      *plugin_info,
                        GObject             *extension,
                        gpointer             user_data)
{
  GPtrArray *ar = user_data;

  g_ptr_array_add (ar, g_object_ref (extension));
}

static GPtrArray *
collect_by_priority (FoundryTextSettings *self)
{
  GPtrArray *ar = g_ptr_array_new_with_free_func (g_object_unref);
  if (self->addins != NULL)
    foundry_extension_set_foreach_by_priority (self->addins, collect_by_priority_cb, ar);
  return ar;
}

static char *
dup_string (FoundryTextSettings *self,
            FoundryTextSetting   setting,
            guint                prop_id)
{
  g_autoptr(GPtrArray) ar = collect_by_priority (self);
  g_auto(GValue) value = G_VALUE_INIT;

  g_value_init (&value, G_TYPE_STRING);

  for (guint i = 0; i < ar->len; i++)
    {
      FoundryTextSettingsProvider *provider = g_ptr_array_index (ar, i);

      if (foundry_text_settings_provider_get_setting (provider, setting, &value))
        return g_value_dup_string (&value);
    }

  return g_value_dup_string (g_param_spec_get_default_value (properties[prop_id]));
}

static gboolean
get_boolean (FoundryTextSettings *self,
             FoundryTextSetting   setting,
             guint                prop_id)
{
  g_autoptr(GPtrArray) ar = collect_by_priority (self);
  g_auto(GValue) value = G_VALUE_INIT;

  g_value_init (&value, G_TYPE_BOOLEAN);

  for (guint i = 0; i < ar->len; i++)
    {
      FoundryTextSettingsProvider *provider = g_ptr_array_index (ar, i);

      if (foundry_text_settings_provider_get_setting (provider, setting, &value))
        return g_value_get_boolean (&value);
    }

  return g_value_get_boolean (g_param_spec_get_default_value (properties[prop_id]));
}

static guint
get_uint (FoundryTextSettings *self,
          FoundryTextSetting   setting,
          guint                prop_id)
{
  g_autoptr(GPtrArray) ar = collect_by_priority (self);
  g_auto(GValue) value = G_VALUE_INIT;

  g_value_init (&value, G_TYPE_UINT);

  for (guint i = 0; i < ar->len; i++)
    {
      FoundryTextSettingsProvider *provider = g_ptr_array_index (ar, i);

      if (foundry_text_settings_provider_get_setting (provider, setting, &value))
        return g_value_get_uint (&value);
    }

  return g_value_get_uint (g_param_spec_get_default_value (properties[prop_id]));
}

static double
get_double (FoundryTextSettings *self,
            FoundryTextSetting   setting,
            guint                prop_id)
{
  g_autoptr(GPtrArray) ar = collect_by_priority (self);
  g_auto(GValue) value = G_VALUE_INIT;

  g_value_init (&value, G_TYPE_DOUBLE);

  for (guint i = 0; i < ar->len; i++)
    {
      FoundryTextSettingsProvider *provider = g_ptr_array_index (ar, i);

      if (foundry_text_settings_provider_get_setting (provider, setting, &value))
        return g_value_get_double (&value);
    }

  return g_value_get_double (g_param_spec_get_default_value (properties[prop_id]));
}

static guint
get_enum (FoundryTextSettings *self,
          FoundryTextSetting   setting,
          guint                prop_id,
          GType                enum_type)
{
  g_autoptr(GPtrArray) ar = collect_by_priority (self);
  g_auto(GValue) value = G_VALUE_INIT;

  g_value_init (&value, enum_type);

  for (guint i = 0; i < ar->len; i++)
    {
      FoundryTextSettingsProvider *provider = g_ptr_array_index (ar, i);

      if (foundry_text_settings_provider_get_setting (provider, setting, &value))
        return g_value_get_enum (&value);
    }

  return g_value_get_enum (g_param_spec_get_default_value (properties[prop_id]));
}

static GParamSpec *
setting_to_param_spec (FoundryTextSetting setting)
{
  switch (setting)
    {
    default:
    case FOUNDRY_TEXT_SETTING_NONE:
      return NULL;

    case FOUNDRY_TEXT_SETTING_AUTO_INDENT:
      return properties[PROP_AUTO_INDENT];

    case FOUNDRY_TEXT_SETTING_ENABLE_SPELL_CHECK:
      return properties[PROP_ENABLE_SPELL_CHECK];

    case FOUNDRY_TEXT_SETTING_ENABLE_COMPLETION:
      return properties[PROP_ENABLE_COMPLETION];

    case FOUNDRY_TEXT_SETTING_ENABLE_SNIPPETS:
      return properties[PROP_ENABLE_SNIPPETS];

    case FOUNDRY_TEXT_SETTING_HIGHLIGHT_CURRENT_LINE:
      return properties[PROP_HIGHLIGHT_CURRENT_LINE];

    case FOUNDRY_TEXT_SETTING_HIGHLIGHT_MATCHING_BRACKETS:
      return properties[PROP_HIGHLIGHT_MATCHING_BRACKETS];

    case FOUNDRY_TEXT_SETTING_IMPLICIT_TRAILING_NEWLINE:
      return properties[PROP_IMPLICIT_TRAILING_NEWLINE];

    case FOUNDRY_TEXT_SETTING_INDENT_ON_TAB:
      return properties[PROP_INDENT_ON_TAB];

    case FOUNDRY_TEXT_SETTING_INSERT_SPACES_INSTEAD_OF_TABS:
      return properties[PROP_INSERT_SPACES_INSTEAD_OF_TABS];

    case FOUNDRY_TEXT_SETTING_INSERT_MATCHING_BRACE:
      return properties[PROP_INSERT_MATCHING_BRACE];

    case FOUNDRY_TEXT_SETTING_LINE_HEIGHT:
      return properties[PROP_LINE_HEIGHT];

    case FOUNDRY_TEXT_SETTING_OVERRIDE_INDENT_WIDTH:
      return properties[PROP_OVERRIDE_INDENT_WIDTH];

    case FOUNDRY_TEXT_SETTING_OVERWRITE_MATCHING_BRACE:
      return properties[PROP_OVERWRITE_MATCHING_BRACE];

    case FOUNDRY_TEXT_SETTING_SHOW_DIAGNOSTICS:
      return properties[PROP_SHOW_DIAGNOSTICS];

    case FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES:
      return properties[PROP_SHOW_LINE_CHANGES];

    case FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES_OVERVIEW:
      return properties[PROP_SHOW_LINE_CHANGES_OVERVIEW];

    case FOUNDRY_TEXT_SETTING_SHOW_LINE_NUMBERS:
      return properties[PROP_SHOW_LINE_NUMBERS];

    case FOUNDRY_TEXT_SETTING_SHOW_RIGHT_MARGIN:
      return properties[PROP_SHOW_RIGHT_MARGIN];

    case FOUNDRY_TEXT_SETTING_SMART_BACKSPACE:
      return properties[PROP_SMART_BACKSPACE];

    case FOUNDRY_TEXT_SETTING_SMART_HOME_END:
      return properties[PROP_SMART_HOME_END];

    case FOUNDRY_TEXT_SETTING_RIGHT_MARGIN_POSITION:
      return properties[PROP_RIGHT_MARGIN_POSITION];

    case FOUNDRY_TEXT_SETTING_TAB_WIDTH:
      return properties[PROP_TAB_WIDTH];

    case FOUNDRY_TEXT_SETTING_INDENT_WIDTH:
      return properties[PROP_INDENT_WIDTH];

    case FOUNDRY_TEXT_SETTING_CUSTOM_FONT:
      return properties[PROP_CUSTOM_FONT];

    case FOUNDRY_TEXT_SETTING_USE_CUSTOM_FONT:
      return properties[PROP_USE_CUSTOM_FONT];

    case FOUNDRY_TEXT_SETTING_COMPLETION_AUTO_SELECT:
      return properties[PROP_COMPLETION_AUTO_SELECT];

    case FOUNDRY_TEXT_SETTING_COMPLETION_PAGE_SIZE:
      return properties[PROP_COMPLETION_PAGE_SIZE];

    case FOUNDRY_TEXT_SETTING_WRAP:
      return properties[PROP_WRAP];
    }
}

static void
foundry_text_settings_provider_changed_cb (FoundryTextSettings         *self,
                                           FoundryTextSetting           setting,
                                           FoundryTextSettingsProvider *provider)
{
  GParamSpec *pspec;

  g_assert (FOUNDRY_IS_TEXT_SETTINGS (self));
  g_assert (FOUNDRY_IS_TEXT_SETTINGS_PROVIDER (provider));

  if ((pspec = setting_to_param_spec (setting)))
    {
      g_object_notify_by_pspec (G_OBJECT (self), pspec);
      return;
    }

  for (guint i = 1; i < N_PROPS; i++)
    g_object_notify_by_pspec (G_OBJECT (self), properties[i]);
}

static void
foundry_text_settings_dispose (GObject *object)
{
  FoundryTextSettings *self = (FoundryTextSettings *)object;

  g_clear_pointer (&self->custom_font, g_free);
  g_clear_object (&self->addins);
  g_weak_ref_set (&self->document_wr, NULL);

  G_OBJECT_CLASS (foundry_text_settings_parent_class)->dispose (object);
}

static void
foundry_text_settings_finalize (GObject *object)
{
  FoundryTextSettings *self = (FoundryTextSettings *)object;

  g_weak_ref_clear (&self->document_wr);

  G_OBJECT_CLASS (foundry_text_settings_parent_class)->finalize (object);
}

static void
foundry_text_settings_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundryTextSettings *self = FOUNDRY_TEXT_SETTINGS (object);

  switch (prop_id)
    {
    case PROP_AUTO_INDENT:
      g_value_set_boolean (value, foundry_text_settings_get_auto_indent (self));
      break;

    case PROP_COMPLETION_AUTO_SELECT:
      g_value_set_boolean (value, foundry_text_settings_get_completion_auto_select (self));
      break;

    case PROP_COMPLETION_PAGE_SIZE:
      g_value_set_uint (value, foundry_text_settings_get_completion_page_size (self));
      break;

    case PROP_ENABLE_SPELL_CHECK:
      g_value_set_boolean (value, foundry_text_settings_get_enable_spell_check (self));
      break;

    case PROP_ENABLE_COMPLETION:
      g_value_set_boolean (value, foundry_text_settings_get_enable_completion (self));
      break;

    case PROP_ENABLE_SNIPPETS:
      g_value_set_boolean (value, foundry_text_settings_get_enable_snippets (self));
      break;

    case PROP_HIGHLIGHT_CURRENT_LINE:
      g_value_set_boolean (value, foundry_text_settings_get_highlight_current_line (self));
      break;

    case PROP_HIGHLIGHT_MATCHING_BRACKETS:
      g_value_set_boolean (value, foundry_text_settings_get_highlight_matching_brackets (self));
      break;

    case PROP_IMPLICIT_TRAILING_NEWLINE:
      g_value_set_boolean (value, foundry_text_settings_get_implicit_trailing_newline (self));
      break;

    case PROP_INDENT_ON_TAB:
      g_value_set_boolean (value, foundry_text_settings_get_indent_on_tab (self));
      break;

    case PROP_INSERT_SPACES_INSTEAD_OF_TABS:
      g_value_set_boolean (value, foundry_text_settings_get_insert_spaces_instead_of_tabs (self));
      break;

    case PROP_INSERT_MATCHING_BRACE:
      g_value_set_boolean (value, foundry_text_settings_get_insert_matching_brace (self));
      break;

    case PROP_LINE_HEIGHT:
      g_value_set_double (value, foundry_text_settings_get_line_height (self));
      break;

    case PROP_OVERRIDE_INDENT_WIDTH:
      g_value_set_boolean (value, foundry_text_settings_get_override_indent_width (self));
      break;

    case PROP_OVERWRITE_MATCHING_BRACE:
      g_value_set_boolean (value, foundry_text_settings_get_overwrite_matching_brace (self));
      break;

    case PROP_SHOW_DIAGNOSTICS:
      g_value_set_boolean (value, foundry_text_settings_get_show_diagnostics (self));
      break;

    case PROP_SHOW_LINE_CHANGES:
      g_value_set_boolean (value, foundry_text_settings_get_show_line_changes (self));
      break;

    case PROP_SHOW_LINE_CHANGES_OVERVIEW:
      g_value_set_boolean (value, foundry_text_settings_get_show_line_changes_overview (self));
      break;

    case PROP_SHOW_LINE_NUMBERS:
      g_value_set_boolean (value, foundry_text_settings_get_show_line_numbers (self));
      break;

    case PROP_SHOW_RIGHT_MARGIN:
      g_value_set_boolean (value, foundry_text_settings_get_show_right_margin (self));
      break;

    case PROP_SMART_BACKSPACE:
      g_value_set_boolean (value, foundry_text_settings_get_smart_backspace (self));
      break;

    case PROP_SMART_HOME_END:
      g_value_set_boolean (value, foundry_text_settings_get_smart_home_end (self));
      break;

    case PROP_RIGHT_MARGIN_POSITION:
      g_value_set_uint (value, foundry_text_settings_get_right_margin_position (self));
      break;

    case PROP_TAB_WIDTH:
      g_value_set_uint (value, foundry_text_settings_get_tab_width (self));
      break;

    case PROP_USE_CUSTOM_FONT:
      g_value_set_boolean (value, foundry_text_settings_get_use_custom_font (self));
      break;

    case PROP_CUSTOM_FONT:
      g_value_take_string (value, foundry_text_settings_dup_custom_font (self));
      break;

    case PROP_INDENT_WIDTH:
      g_value_set_uint (value, foundry_text_settings_get_indent_width (self));
      break;

    case PROP_DOCUMENT:
      g_value_take_object (value, g_weak_ref_get (&self->document_wr));
      break;

    case PROP_WRAP:
      g_value_set_enum (value, foundry_text_settings_get_wrap (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_text_settings_set_property (GObject      *object,
                                    guint         prop_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
  FoundryTextSettings *self = FOUNDRY_TEXT_SETTINGS (object);

  switch (prop_id)
    {
    case PROP_AUTO_INDENT:
      foundry_text_settings_set_auto_indent (self, g_value_get_boolean (value));
      break;

    case PROP_COMPLETION_AUTO_SELECT:
      foundry_text_settings_set_completion_auto_select (self, g_value_get_boolean (value));
      break;

    case PROP_COMPLETION_PAGE_SIZE:
      foundry_text_settings_set_completion_page_size (self, g_value_get_uint (value));
      break;

    case PROP_ENABLE_SPELL_CHECK:
      foundry_text_settings_set_enable_spell_check (self, g_value_get_boolean (value));
      break;

    case PROP_ENABLE_COMPLETION:
      foundry_text_settings_set_enable_completion (self, g_value_get_boolean (value));
      break;

    case PROP_ENABLE_SNIPPETS:
      foundry_text_settings_set_enable_snippets (self, g_value_get_boolean (value));
      break;

    case PROP_HIGHLIGHT_CURRENT_LINE:
      foundry_text_settings_set_highlight_current_line (self, g_value_get_boolean (value));
      break;

    case PROP_HIGHLIGHT_MATCHING_BRACKETS:
      foundry_text_settings_set_highlight_matching_brackets (self, g_value_get_boolean (value));
      break;

    case PROP_IMPLICIT_TRAILING_NEWLINE:
      foundry_text_settings_set_implicit_trailing_newline (self, g_value_get_boolean (value));
      break;

    case PROP_INDENT_ON_TAB:
      foundry_text_settings_set_indent_on_tab (self, g_value_get_boolean (value));
      break;

    case PROP_INSERT_MATCHING_BRACE:
      foundry_text_settings_set_insert_matching_brace (self, g_value_get_boolean (value));
      break;

    case PROP_INSERT_SPACES_INSTEAD_OF_TABS:
      foundry_text_settings_set_insert_spaces_instead_of_tabs (self, g_value_get_boolean (value));
      break;

    case PROP_LINE_HEIGHT:
      foundry_text_settings_set_line_height (self, g_value_get_double (value));
      break;

    case PROP_OVERRIDE_INDENT_WIDTH:
      foundry_text_settings_set_override_indent_width (self, g_value_get_boolean (value));
      break;

    case PROP_OVERWRITE_MATCHING_BRACE:
      foundry_text_settings_set_overwrite_matching_brace (self, g_value_get_boolean (value));
      break;

    case PROP_RIGHT_MARGIN_POSITION:
      foundry_text_settings_set_right_margin_position (self, g_value_get_uint (value));
      break;

    case PROP_SHOW_DIAGNOSTICS:
      foundry_text_settings_set_show_diagnostics (self, g_value_get_boolean (value));
      break;

    case PROP_SHOW_LINE_CHANGES:
      foundry_text_settings_set_show_line_changes (self, g_value_get_boolean (value));
      break;

    case PROP_SHOW_LINE_CHANGES_OVERVIEW:
      foundry_text_settings_set_show_line_changes_overview (self, g_value_get_boolean (value));
      break;

    case PROP_SHOW_LINE_NUMBERS:
      foundry_text_settings_set_show_line_numbers (self, g_value_get_boolean (value));
      break;

    case PROP_SHOW_RIGHT_MARGIN:
      foundry_text_settings_set_show_right_margin (self, g_value_get_boolean (value));
      break;

    case PROP_SMART_BACKSPACE:
      foundry_text_settings_set_smart_backspace (self, g_value_get_boolean (value));
      break;

    case PROP_SMART_HOME_END:
      foundry_text_settings_set_smart_home_end (self, g_value_get_boolean (value));
      break;

    case PROP_TAB_WIDTH:
      foundry_text_settings_set_tab_width (self, g_value_get_uint (value));
      break;

    case PROP_USE_CUSTOM_FONT:
      foundry_text_settings_set_use_custom_font (self, g_value_get_boolean (value));
      break;

    case PROP_CUSTOM_FONT:
      foundry_text_settings_set_custom_font (self, g_value_get_string (value));
      break;

    case PROP_INDENT_WIDTH:
      foundry_text_settings_set_indent_width (self, g_value_get_uint (value));
      break;

    case PROP_WRAP:
      foundry_text_settings_set_wrap (self, g_value_get_enum (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_text_settings_class_init (FoundryTextSettingsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_text_settings_dispose;
  object_class->finalize = foundry_text_settings_finalize;
  object_class->get_property = foundry_text_settings_get_property;
  object_class->set_property = foundry_text_settings_set_property;

  properties[PROP_AUTO_INDENT] =
    g_param_spec_boolean ("auto-indent", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_COMPLETION_AUTO_SELECT] =
    g_param_spec_boolean ("completion-auto-select", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_COMPLETION_PAGE_SIZE] =
    g_param_spec_uint ("completion-page-size", NULL, NULL,
                       1, 32, 5,
                       (G_PARAM_READWRITE |
                        G_PARAM_EXPLICIT_NOTIFY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_ENABLE_COMPLETION] =
    g_param_spec_boolean ("enable-completion", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_ENABLE_SNIPPETS] =
    g_param_spec_boolean ("enable-snippets", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_ENABLE_SPELL_CHECK] =
    g_param_spec_boolean ("enable-spell-check", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_HIGHLIGHT_CURRENT_LINE] =
    g_param_spec_boolean ("highlight-current-line", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_HIGHLIGHT_MATCHING_BRACKETS] =
    g_param_spec_boolean ("highlight-matching-brackets", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_IMPLICIT_TRAILING_NEWLINE] =
    g_param_spec_boolean ("implicit-trailing-newline", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_INDENT_ON_TAB] =
    g_param_spec_boolean ("indent-on-tab", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_INSERT_SPACES_INSTEAD_OF_TABS] =
    g_param_spec_boolean ("insert-spaces-instead-of-tabs", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_INSERT_MATCHING_BRACE] =
    g_param_spec_boolean ("insert-matching-brace", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_LINE_HEIGHT] =
    g_param_spec_double ("line-height", NULL, NULL,
                         .1, 10., 1.1,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_OVERRIDE_INDENT_WIDTH] =
    g_param_spec_boolean ("override-indent-width", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_OVERWRITE_MATCHING_BRACE] =
    g_param_spec_boolean ("overwrite-matching-brace", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SHOW_DIAGNOSTICS] =
    g_param_spec_boolean ("show-diagnostics", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SHOW_LINE_CHANGES] =
    g_param_spec_boolean ("show-line-changes", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SHOW_LINE_CHANGES_OVERVIEW] =
    g_param_spec_boolean ("show-line-changes-overview", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SHOW_LINE_NUMBERS] =
    g_param_spec_boolean ("show-line-numbers", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SHOW_RIGHT_MARGIN] =
    g_param_spec_boolean ("show-right-margin", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SMART_BACKSPACE] =
    g_param_spec_boolean ("smart-backspace", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SMART_HOME_END] =
    g_param_spec_boolean ("smart-home-end", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_RIGHT_MARGIN_POSITION] =
    g_param_spec_uint ("right-margin-position", NULL, NULL,
                       1, 1000, 80,
                       (G_PARAM_READWRITE |
                        G_PARAM_EXPLICIT_NOTIFY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_TAB_WIDTH] =
    g_param_spec_uint ("tab-width", NULL, NULL,
                       1, 32, 8,
                       (G_PARAM_READWRITE |
                        G_PARAM_EXPLICIT_NOTIFY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_INDENT_WIDTH] =
    g_param_spec_uint ("indent-width", NULL, NULL,
                       1, 32, 8,
                       (G_PARAM_READWRITE |
                        G_PARAM_EXPLICIT_NOTIFY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_CUSTOM_FONT] =
    g_param_spec_string ("custom-font", NULL, NULL,
                         "Monospace 11",
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_USE_CUSTOM_FONT] =
    g_param_spec_boolean ("use-custom-font", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_DOCUMENT] =
    g_param_spec_object ("document", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_DOCUMENT,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_WRAP] =
    g_param_spec_enum ("wrap", NULL, NULL,
                       FOUNDRY_TYPE_TEXT_WRAP,
                       FOUNDRY_TEXT_WRAP_NONE,
                       (G_PARAM_READWRITE |
                        G_PARAM_EXPLICIT_NOTIFY |
                        G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_text_settings_init (FoundryTextSettings *self)
{
  g_weak_ref_init (&self->document_wr, NULL);

  self->highlight_matching_brackets = TRUE;
  self->right_margin_position = 80;
  self->indent_width = -1;
  self->show_line_changes = TRUE;
  self->show_line_numbers = TRUE;
  self->tab_width = 80;
  self->wrap = FOUNDRY_TEXT_WRAP_NONE;
}

static void
foundry_text_settings_provider_added_cb (FoundryExtensionSet *set,
                                         PeasPluginInfo      *plugin_info,
                                         GObject             *extension,
                                         gpointer             user_data)
{
  FoundryTextSettings *self = user_data;
  FoundryTextSettingsProvider *provider = (FoundryTextSettingsProvider *)extension;
  g_autoptr(FoundryTextDocument) document = NULL;

  g_assert (FOUNDRY_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_TEXT_SETTINGS_PROVIDER (provider));
  g_assert (FOUNDRY_IS_TEXT_SETTINGS (self));

  document = g_weak_ref_get (&self->document_wr);

  g_signal_connect_object (provider,
                           "changed",
                           G_CALLBACK (foundry_text_settings_provider_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  dex_future_disown (_foundry_text_settings_provider_load (provider, document));
}

static void
foundry_text_settings_provider_removed_cb (FoundryExtensionSet *set,
                                           PeasPluginInfo      *plugin_info,
                                           GObject             *extension,
                                           gpointer             user_data)
{
  FoundryTextSettingsProvider *provider = (FoundryTextSettingsProvider *)extension;

  g_assert (FOUNDRY_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_TEXT_SETTINGS_PROVIDER (provider));
  g_assert (FOUNDRY_IS_TEXT_SETTINGS (user_data));

  dex_future_disown (_foundry_text_settings_provider_unload (provider));
}

DexFuture *
_foundry_text_settings_new (FoundryTextDocument *document)
{
  g_autoptr(FoundryTextSettings) self = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  g_return_val_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (document), NULL);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (document));

  self = g_object_new (FOUNDRY_TYPE_TEXT_SETTINGS,
                       "context", context,
                       NULL);

  self->addins = foundry_extension_set_new (context,
                                            peas_engine_get_default (),
                                            FOUNDRY_TYPE_TEXT_SETTINGS_PROVIDER,
                                            "Text-Settings-Provider", "*",
                                            NULL);

  g_signal_connect (self->addins,
                    "extension-added",
                    G_CALLBACK (foundry_text_settings_provider_added_cb),
                    self);
  g_signal_connect (self->addins,
                    "extension-removed",
                    G_CALLBACK (foundry_text_settings_provider_removed_cb),
                    self);

  futures = g_ptr_array_new_with_free_func (dex_unref);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTextSettingsProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_signal_connect_object (provider,
                               "changed",
                               G_CALLBACK (foundry_text_settings_provider_changed_cb),
                               self,
                               G_CONNECT_SWAPPED);

      g_ptr_array_add (futures, _foundry_text_settings_provider_load (provider, document));
    }

  if (futures->len == 0)
    return dex_future_new_take_object (g_steal_pointer (&self));

  return dex_future_finally (foundry_future_all (futures),
                             foundry_future_return_object,
                             g_object_ref (self),
                             g_object_unref);
}


gboolean
foundry_text_settings_get_auto_indent (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->auto_indent_set)
    return self->auto_indent;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_AUTO_INDENT, PROP_AUTO_INDENT);
}

void
foundry_text_settings_set_auto_indent (FoundryTextSettings *self,
                                       gboolean             auto_indent)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  auto_indent = !!auto_indent;

  if (auto_indent != self->auto_indent)
    {
      self->auto_indent = auto_indent;
      self->auto_indent_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_AUTO_INDENT]);
    }
}

gboolean
foundry_text_settings_get_enable_completion (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->enable_completion_set)
    return self->enable_completion;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_ENABLE_COMPLETION, PROP_ENABLE_COMPLETION);
}

void
foundry_text_settings_set_enable_completion (FoundryTextSettings *self,
                                             gboolean             enable_completion)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  enable_completion = !!enable_completion;

  if (enable_completion != self->enable_completion)
    {
      self->enable_completion = enable_completion;
      self->enable_completion_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ENABLE_COMPLETION]);
    }
}

gboolean
foundry_text_settings_get_enable_snippets (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->enable_snippets_set)
    return self->enable_snippets;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_ENABLE_SNIPPETS, PROP_ENABLE_SNIPPETS);
}

void
foundry_text_settings_set_enable_snippets (FoundryTextSettings *self,
                                           gboolean             enable_snippets)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  enable_snippets = !!enable_snippets;

  if (enable_snippets != self->enable_snippets)
    {
      self->enable_snippets = enable_snippets;
      self->enable_snippets_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ENABLE_SNIPPETS]);
    }
}

gboolean
foundry_text_settings_get_enable_spell_check (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->enable_spell_check_set)
    return self->enable_spell_check;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_ENABLE_SPELL_CHECK, PROP_ENABLE_SPELL_CHECK);
}

void
foundry_text_settings_set_enable_spell_check (FoundryTextSettings *self,
                                              gboolean             enable_spell_check)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  enable_spell_check = !!enable_spell_check;

  if (enable_spell_check != self->enable_spell_check)
    {
      self->enable_spell_check = enable_spell_check;
      self->enable_spell_check_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ENABLE_SPELL_CHECK]);
    }
}

gboolean
foundry_text_settings_get_highlight_current_line (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->highlight_current_line_set)
    return self->highlight_current_line;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_HIGHLIGHT_CURRENT_LINE, PROP_HIGHLIGHT_CURRENT_LINE);
}

void
foundry_text_settings_set_highlight_current_line (FoundryTextSettings *self,
                                                  gboolean             highlight_current_line)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  highlight_current_line = !!highlight_current_line;

  if (highlight_current_line != self->highlight_current_line)
    {
      self->highlight_current_line = highlight_current_line;
      self->highlight_current_line_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_HIGHLIGHT_CURRENT_LINE]);
    }
}

gboolean
foundry_text_settings_get_highlight_matching_brackets (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->highlight_matching_brackets_set)
    return self->highlight_matching_brackets;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_HIGHLIGHT_MATCHING_BRACKETS, PROP_HIGHLIGHT_MATCHING_BRACKETS);
}

void
foundry_text_settings_set_highlight_matching_brackets (FoundryTextSettings *self,
                                                       gboolean             highlight_matching_brackets)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  highlight_matching_brackets = !!highlight_matching_brackets;

  if (highlight_matching_brackets != self->highlight_matching_brackets)
    {
      self->highlight_matching_brackets = highlight_matching_brackets;
      self->highlight_matching_brackets_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_HIGHLIGHT_MATCHING_BRACKETS]);
    }
}

gboolean
foundry_text_settings_get_implicit_trailing_newline (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->implicit_trailing_newline_set)
    return self->implicit_trailing_newline;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_IMPLICIT_TRAILING_NEWLINE, PROP_IMPLICIT_TRAILING_NEWLINE);
}

void
foundry_text_settings_set_implicit_trailing_newline (FoundryTextSettings *self,
                                                     gboolean             implicit_trailing_newline)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  implicit_trailing_newline = !!implicit_trailing_newline;

  if (implicit_trailing_newline != self->implicit_trailing_newline)
    {
      self->implicit_trailing_newline = implicit_trailing_newline;
      self->implicit_trailing_newline_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_IMPLICIT_TRAILING_NEWLINE]);
    }
}

gboolean
foundry_text_settings_get_indent_on_tab (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->indent_on_tab_set)
    return self->indent_on_tab;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_INDENT_ON_TAB, PROP_INDENT_ON_TAB);
}

void
foundry_text_settings_set_indent_on_tab (FoundryTextSettings *self,
                                         gboolean             indent_on_tab)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  indent_on_tab = !!indent_on_tab;

  if (indent_on_tab != self->indent_on_tab)
    {
      self->indent_on_tab = indent_on_tab;
      self->indent_on_tab_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_INDENT_ON_TAB]);
    }
}

guint
foundry_text_settings_get_indent_width (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), -1);

  if (self->indent_width_set)
    return self->indent_width;

  return get_uint (self, FOUNDRY_TEXT_SETTING_INDENT_WIDTH, PROP_INDENT_WIDTH);
}

void
foundry_text_settings_set_indent_width (FoundryTextSettings *self,
                                        guint                indent_width)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));
  g_return_if_fail (indent_width > 0);
  g_return_if_fail (indent_width <= 32);

  indent_width = !!indent_width;

  if (indent_width != self->indent_width)
    {
      self->indent_width = indent_width;
      self->indent_width_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_INDENT_WIDTH]);
    }
}

gboolean
foundry_text_settings_get_insert_matching_brace (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->insert_matching_brace_set)
    return self->insert_matching_brace;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_INSERT_MATCHING_BRACE, PROP_INSERT_MATCHING_BRACE);
}

void
foundry_text_settings_set_insert_matching_brace (FoundryTextSettings *self,
                                                 gboolean             insert_matching_brace)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  insert_matching_brace = !!insert_matching_brace;

  if (insert_matching_brace != self->insert_matching_brace)
    {
      self->insert_matching_brace = insert_matching_brace;
      self->insert_matching_brace_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_INSERT_MATCHING_BRACE]);
    }
}

gboolean
foundry_text_settings_get_insert_spaces_instead_of_tabs (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->insert_spaces_instead_of_tabs_set)
    return self->insert_spaces_instead_of_tabs;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_INSERT_SPACES_INSTEAD_OF_TABS, PROP_INSERT_SPACES_INSTEAD_OF_TABS);
}

void
foundry_text_settings_set_insert_spaces_instead_of_tabs (FoundryTextSettings *self,
                                                         gboolean             insert_spaces_instead_of_tabs)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  insert_spaces_instead_of_tabs = !!insert_spaces_instead_of_tabs;

  if (insert_spaces_instead_of_tabs != self->insert_spaces_instead_of_tabs)
    {
      self->insert_spaces_instead_of_tabs = insert_spaces_instead_of_tabs;
      self->insert_spaces_instead_of_tabs_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_INSERT_SPACES_INSTEAD_OF_TABS]);
    }
}

double
foundry_text_settings_get_line_height (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), 1.);

  if (self->line_height_set)
    return self->line_height;

  return get_double (self, FOUNDRY_TEXT_SETTING_LINE_HEIGHT, PROP_LINE_HEIGHT);
}

void
foundry_text_settings_set_line_height (FoundryTextSettings *self,
                                       double               line_height)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  line_height = CLAMP (line_height, .1, 10.);

  if (line_height != self->line_height)
    {
      self->line_height = line_height;
      self->line_height_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_LINE_HEIGHT]);
    }
}

gboolean
foundry_text_settings_get_override_indent_width (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->override_indent_width_set)
    return self->override_indent_width;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_OVERRIDE_INDENT_WIDTH, PROP_OVERRIDE_INDENT_WIDTH);
}

void
foundry_text_settings_set_override_indent_width (FoundryTextSettings *self,
                                                 gboolean             override_indent_width)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  override_indent_width = !!override_indent_width;

  if (override_indent_width != self->override_indent_width)
    {
      self->override_indent_width = override_indent_width;
      self->override_indent_width_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_OVERRIDE_INDENT_WIDTH]);
    }
}

gboolean
foundry_text_settings_get_overwrite_matching_brace (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->overwrite_matching_brace_set)
    return self->overwrite_matching_brace;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_OVERWRITE_MATCHING_BRACE, PROP_OVERWRITE_MATCHING_BRACE);
}

void
foundry_text_settings_set_overwrite_matching_brace (FoundryTextSettings *self,
                                                    gboolean             overwrite_matching_brace)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  overwrite_matching_brace = !!overwrite_matching_brace;

  if (overwrite_matching_brace != self->overwrite_matching_brace)
    {
      self->overwrite_matching_brace = overwrite_matching_brace;
      self->overwrite_matching_brace_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_OVERWRITE_MATCHING_BRACE]);
    }
}

guint
foundry_text_settings_get_right_margin_position (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), 80);

  if (self->right_margin_position_set)
    return self->right_margin_position;

  return get_uint (self, FOUNDRY_TEXT_SETTING_RIGHT_MARGIN_POSITION, PROP_RIGHT_MARGIN_POSITION);
}

void
foundry_text_settings_set_right_margin_position (FoundryTextSettings *self,
                                                 guint                right_margin_position)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));
  g_return_if_fail (right_margin_position > 0);
  g_return_if_fail (right_margin_position <= 1000);

  right_margin_position = !!right_margin_position;

  if (right_margin_position != self->right_margin_position)
    {
      self->right_margin_position = right_margin_position;
      self->right_margin_position_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_RIGHT_MARGIN_POSITION]);
    }
}

gboolean
foundry_text_settings_get_show_diagnostics (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->show_diagnostics_set)
    return self->show_diagnostics;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_SHOW_DIAGNOSTICS, PROP_SHOW_DIAGNOSTICS);
}

void
foundry_text_settings_set_show_diagnostics (FoundryTextSettings *self,
                                            gboolean             show_diagnostics)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  show_diagnostics = !!show_diagnostics;

  if (show_diagnostics != self->show_diagnostics)
    {
      self->show_diagnostics = show_diagnostics;
      self->show_diagnostics_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_DIAGNOSTICS]);
    }
}

gboolean
foundry_text_settings_get_show_line_changes (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->show_line_changes_set)
    return self->show_line_changes;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES, PROP_SHOW_LINE_CHANGES);
}

void
foundry_text_settings_set_show_line_changes (FoundryTextSettings *self,
                                             gboolean             show_line_changes)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  show_line_changes = !!show_line_changes;

  if (show_line_changes != self->show_line_changes)
    {
      self->show_line_changes = show_line_changes;
      self->show_line_changes_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_LINE_CHANGES]);
    }
}

gboolean
foundry_text_settings_get_show_line_changes_overview (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->show_line_changes_overview_set)
    return self->show_line_changes_overview;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES_OVERVIEW, PROP_SHOW_LINE_CHANGES_OVERVIEW);
}

void
foundry_text_settings_set_show_line_changes_overview (FoundryTextSettings *self,
                                                      gboolean             show_line_changes_overview)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  show_line_changes_overview = !!show_line_changes_overview;

  if (show_line_changes_overview != self->show_line_changes_overview)
    {
      self->show_line_changes_overview = show_line_changes_overview;
      self->show_line_changes_overview_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_LINE_CHANGES_OVERVIEW]);
    }
}

gboolean
foundry_text_settings_get_show_line_numbers (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->show_line_numbers_set)
    return self->show_line_numbers;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_SHOW_LINE_NUMBERS, PROP_SHOW_LINE_NUMBERS);
}

void
foundry_text_settings_set_show_line_numbers (FoundryTextSettings *self,
                                             gboolean             show_line_numbers)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  show_line_numbers = !!show_line_numbers;

  if (show_line_numbers != self->show_line_numbers)
    {
      self->show_line_numbers = show_line_numbers;
      self->show_line_numbers_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_LINE_NUMBERS]);
    }
}

gboolean
foundry_text_settings_get_show_right_margin (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->show_right_margin_set)
    return self->show_right_margin;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_SHOW_RIGHT_MARGIN, PROP_SHOW_RIGHT_MARGIN);
}

void
foundry_text_settings_set_show_right_margin (FoundryTextSettings *self,
                                             gboolean             show_right_margin)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  show_right_margin = !!show_right_margin;

  if (show_right_margin != self->show_right_margin)
    {
      self->show_right_margin = show_right_margin;
      self->show_right_margin_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_RIGHT_MARGIN]);
    }
}

gboolean
foundry_text_settings_get_smart_backspace (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->smart_backspace_set)
    return self->smart_backspace;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_SMART_BACKSPACE, PROP_SMART_BACKSPACE);
}

void
foundry_text_settings_set_smart_backspace (FoundryTextSettings *self,
                                           gboolean             smart_backspace)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  smart_backspace = !!smart_backspace;

  if (smart_backspace != self->smart_backspace)
    {
      self->smart_backspace = smart_backspace;
      self->smart_backspace_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SMART_BACKSPACE]);
    }
}

gboolean
foundry_text_settings_get_smart_home_end (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->smart_home_end_set)
    return self->smart_home_end;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_SMART_HOME_END, PROP_SMART_HOME_END);
}

void
foundry_text_settings_set_smart_home_end (FoundryTextSettings *self,
                                          gboolean             smart_home_end)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  smart_home_end = !!smart_home_end;

  if (smart_home_end != self->smart_home_end)
    {
      self->smart_home_end = smart_home_end;
      self->smart_home_end_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SMART_HOME_END]);
    }
}

guint
foundry_text_settings_get_tab_width (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), 0);

  if (self->tab_width_set)
    return self->tab_width;

  return get_uint (self, FOUNDRY_TEXT_SETTING_TAB_WIDTH, PROP_TAB_WIDTH);
}

void
foundry_text_settings_set_tab_width (FoundryTextSettings *self,
                                     guint                tab_width)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));
  g_return_if_fail (tab_width > 0);
  g_return_if_fail (tab_width <= 32);

  if (tab_width != self->tab_width)
    {
      self->tab_width = tab_width;
      self->tab_width_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_TAB_WIDTH]);
    }
}

gboolean
foundry_text_settings_get_use_custom_font (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->use_custom_font_set)
    return self->use_custom_font;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_USE_CUSTOM_FONT, PROP_USE_CUSTOM_FONT);
}

void
foundry_text_settings_set_use_custom_font (FoundryTextSettings *self,
                                           gboolean             use_custom_font)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  use_custom_font = !!use_custom_font;

  if (use_custom_font != self->use_custom_font)
    {
      self->use_custom_font = use_custom_font;
      self->use_custom_font_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_USE_CUSTOM_FONT]);
    }
}

char *
foundry_text_settings_dup_custom_font (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), NULL);

  if (self->custom_font_set)
    return g_strdup (self->custom_font);

  return dup_string (self, FOUNDRY_TEXT_SETTING_CUSTOM_FONT, PROP_CUSTOM_FONT);
}

void
foundry_text_settings_set_custom_font (FoundryTextSettings *self,
                                       const char          *custom_font)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  if (g_set_str (&self->custom_font, custom_font))
    {
      self->custom_font_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CUSTOM_FONT]);
    }
}

gboolean
foundry_text_settings_get_completion_auto_select (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), FALSE);

  if (self->completion_auto_select_set)
    return self->completion_auto_select;

  return get_boolean (self, FOUNDRY_TEXT_SETTING_COMPLETION_AUTO_SELECT, PROP_COMPLETION_AUTO_SELECT);
}

void
foundry_text_settings_set_completion_auto_select (FoundryTextSettings *self,
                                                  gboolean             completion_auto_select)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));

  completion_auto_select = !!completion_auto_select;

  if (completion_auto_select != self->completion_auto_select)
    {
      self->completion_auto_select = completion_auto_select;
      self->completion_auto_select_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_COMPLETION_AUTO_SELECT]);
    }
}

guint
foundry_text_settings_get_completion_page_size (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), -1);

  if (self->completion_page_size_set)
    return self->completion_page_size;

  return get_uint (self, FOUNDRY_TEXT_SETTING_COMPLETION_PAGE_SIZE, PROP_COMPLETION_PAGE_SIZE);
}

void
foundry_text_settings_set_completion_page_size (FoundryTextSettings *self,
                                                guint                completion_page_size)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));
  g_return_if_fail (completion_page_size > 0);
  g_return_if_fail (completion_page_size <= 32);

  completion_page_size = !!completion_page_size;

  if (completion_page_size != self->completion_page_size)
    {
      self->completion_page_size = completion_page_size;
      self->completion_page_size_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_COMPLETION_PAGE_SIZE]);
    }
}

FoundryTextWrap
foundry_text_settings_get_wrap (FoundryTextSettings *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self), 0);

  if (self->wrap_set)
    return self->wrap;

  return get_enum (self, FOUNDRY_TEXT_SETTING_WRAP, PROP_WRAP, FOUNDRY_TYPE_TEXT_WRAP);
}

void
foundry_text_settings_set_wrap (FoundryTextSettings *self,
                                FoundryTextWrap      wrap)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS (self));
  g_return_if_fail (wrap <= FOUNDRY_TEXT_WRAP_WORD_CHAR);

  if (wrap != self->wrap)
    {
      self->wrap = wrap;
      self->wrap_set = TRUE;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_WRAP]);
    }
}
