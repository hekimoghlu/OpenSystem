/* plugin-editorconfig-settings-provider.c
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

#include "editorconfig-glib.h"

#include "plugin-editorconfig-settings-provider.h"

struct _PluginEditorconfigSettingsProvider
{
  FoundryTextSettingsProvider parent_instance;
  GHashTable *ht;
};

G_DEFINE_FINAL_TYPE (PluginEditorconfigSettingsProvider, plugin_editorconfig_settings_provider, FOUNDRY_TYPE_TEXT_SETTINGS_PROVIDER)

static gboolean
plugin_editorconfig_settings_provider_get_setting (FoundryTextSettingsProvider *provider,
                                                   FoundryTextSetting           setting,
                                                   GValue                      *value)
{
  PluginEditorconfigSettingsProvider *self = PLUGIN_EDITORCONFIG_SETTINGS_PROVIDER (provider);
  const GValue *src = NULL;

  if (self->ht == NULL)
    return FALSE;

  switch (setting)
    {
    case FOUNDRY_TEXT_SETTING_OVERRIDE_INDENT_WIDTH:
      if ((src = g_hash_table_lookup (self->ht, "indent_size")))
        {
          g_value_set_boolean (value, TRUE);
          return TRUE;
        }
      break;

    case FOUNDRY_TEXT_SETTING_INDENT_WIDTH:
      src = g_hash_table_lookup (self->ht, "indent_size");
      break;

    case FOUNDRY_TEXT_SETTING_TAB_WIDTH:
      src = g_hash_table_lookup (self->ht, "tab_width");
      break;

    case FOUNDRY_TEXT_SETTING_IMPLICIT_TRAILING_NEWLINE:
      src = g_hash_table_lookup (self->ht, "insert_final_newline");
      break;

    case FOUNDRY_TEXT_SETTING_RIGHT_MARGIN_POSITION:
      src = g_hash_table_lookup (self->ht, "max_line_length");
      break;

    case FOUNDRY_TEXT_SETTING_INSERT_SPACES_INSTEAD_OF_TABS:
      if ((src = g_hash_table_lookup (self->ht, "indent_style")))
        {
          g_value_set_boolean (value, !foundry_str_equal0 (g_value_get_string (src), "tab"));
          return TRUE;
        }
      break;

    case FOUNDRY_TEXT_SETTING_NONE:
    case FOUNDRY_TEXT_SETTING_AUTO_INDENT:
    case FOUNDRY_TEXT_SETTING_COMPLETION_AUTO_SELECT:
    case FOUNDRY_TEXT_SETTING_COMPLETION_PAGE_SIZE:
    case FOUNDRY_TEXT_SETTING_CUSTOM_FONT:
    case FOUNDRY_TEXT_SETTING_ENABLE_COMPLETION:
    case FOUNDRY_TEXT_SETTING_ENABLE_SNIPPETS:
    case FOUNDRY_TEXT_SETTING_ENABLE_SPELL_CHECK:
    case FOUNDRY_TEXT_SETTING_HIGHLIGHT_CURRENT_LINE:
    case FOUNDRY_TEXT_SETTING_HIGHLIGHT_MATCHING_BRACKETS:
    case FOUNDRY_TEXT_SETTING_INDENT_ON_TAB:
    case FOUNDRY_TEXT_SETTING_INSERT_MATCHING_BRACE:
    case FOUNDRY_TEXT_SETTING_LINE_HEIGHT:
    case FOUNDRY_TEXT_SETTING_OVERWRITE_MATCHING_BRACE:
    case FOUNDRY_TEXT_SETTING_SHOW_DIAGNOSTICS:
    case FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES:
    case FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES_OVERVIEW:
    case FOUNDRY_TEXT_SETTING_SHOW_LINE_NUMBERS:
    case FOUNDRY_TEXT_SETTING_SHOW_RIGHT_MARGIN:
    case FOUNDRY_TEXT_SETTING_SMART_BACKSPACE:
    case FOUNDRY_TEXT_SETTING_SMART_HOME_END:
    case FOUNDRY_TEXT_SETTING_USE_CUSTOM_FONT:
    case FOUNDRY_TEXT_SETTING_WRAP:
    default:
      break;
    }

  if (src != NULL)
    {
      if (G_VALUE_HOLDS_INT (src) && G_VALUE_HOLDS_UINT (value))
        {
          if (g_value_get_int (src) < 0)
            return FALSE;
          g_value_set_uint (value, g_value_get_int (src));
        }
      else
        {
          g_value_copy (src, value);
        }

      return TRUE;
    }

  return FALSE;
}

static DexFuture *
plugin_editorconfig_settings_provider_apply (DexFuture *completed,
                                             gpointer   user_data)
{
  PluginEditorconfigSettingsProvider *self = user_data;
  g_autoptr(GHashTable) ht = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_EDITORCONFIG_SETTINGS_PROVIDER (self));
  g_assert (DEX_IS_FUTURE (completed));

  g_clear_pointer (&self->ht, g_hash_table_unref);
  self->ht = dex_await_boxed (dex_ref (completed), NULL);

  foundry_text_settings_provider_emit_changed (FOUNDRY_TEXT_SETTINGS_PROVIDER (self), 0);

  return dex_ref (completed);
}

static DexFuture *
plugin_editorconfig_settings_provider_load_fiber (gpointer user_data)
{
  GFile *file = user_data;
  g_autoptr(GHashTable) ht = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (G_IS_FILE (file));

  if (!(ht = editorconfig_glib_read (file, NULL, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_boxed (G_TYPE_HASH_TABLE, g_steal_pointer (&ht));
}

static DexFuture *
plugin_editorconfig_settings_provider_load (FoundryTextSettingsProvider *provider)
{
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(GFile) file = NULL;

  g_assert (PLUGIN_IS_EDITORCONFIG_SETTINGS_PROVIDER (provider));

  document = foundry_text_settings_provider_dup_document (provider);
  file = foundry_text_document_dup_file (document);

  return dex_future_then (dex_thread_spawn ("[editorconfig-settings]",
                                            plugin_editorconfig_settings_provider_load_fiber,
                                            g_object_ref (file),
                                            g_object_unref),
                          plugin_editorconfig_settings_provider_apply,
                          g_object_ref (provider),
                          g_object_unref);
}

static void
plugin_editorconfig_settings_provider_finalize (GObject *object)
{
  PluginEditorconfigSettingsProvider *self = (PluginEditorconfigSettingsProvider *)object;

  g_clear_pointer (&self->ht, g_hash_table_unref);

  G_OBJECT_CLASS (plugin_editorconfig_settings_provider_parent_class)->finalize (object);
}

static void
plugin_editorconfig_settings_provider_class_init (PluginEditorconfigSettingsProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryTextSettingsProviderClass *provider_class = FOUNDRY_TEXT_SETTINGS_PROVIDER_CLASS (klass);

  provider_class->load = plugin_editorconfig_settings_provider_load;
  provider_class->get_setting = plugin_editorconfig_settings_provider_get_setting;

  object_class->finalize = plugin_editorconfig_settings_provider_finalize;
}

static void
plugin_editorconfig_settings_provider_init (PluginEditorconfigSettingsProvider *self)
{
}
