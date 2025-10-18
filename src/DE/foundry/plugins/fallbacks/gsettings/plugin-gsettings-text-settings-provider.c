/* plugin-gsettings-text-settings-provider.c
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

#include "plugin-gsettings-text-settings-provider.h"

#include "gsettings-mapping.h"

struct _PluginGsettingsTextSettingsProvider
{
  FoundryTextSettingsProvider  parent_instance;
  FoundrySettings             *settings;
};

G_DEFINE_FINAL_TYPE (PluginGsettingsTextSettingsProvider, plugin_gsettings_text_settings_provider, FOUNDRY_TYPE_TEXT_SETTINGS_PROVIDER)

static FoundrySettings *default_settings;

static void
plugin_gsettings_text_settings_provider_changed_cb (PluginGsettingsTextSettingsProvider *self,
                                                    const char                          *key,
                                                    FoundrySettings                     *settings)
{
  GEnumClass *klass = NULL;
  GEnumValue *value;

  g_assert (PLUGIN_IS_GSETTINGS_TEXT_SETTINGS_PROVIDER (self));
  g_assert (key != NULL);
  g_assert (FOUNDRY_IS_SETTINGS (settings));

  klass = g_type_class_get (FOUNDRY_TYPE_TEXT_SETTING);

  if ((value = g_enum_get_value_by_nick (klass, key)))
    foundry_text_settings_provider_emit_changed (FOUNDRY_TEXT_SETTINGS_PROVIDER (self), value->value);
}

static void
plugin_gsettings_text_settings_provider_reload (PluginGsettingsTextSettingsProvider *self)
{
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryTextBuffer) buffer = NULL;
  g_autoptr(FoundrySettings) settings = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autofree char *language_id = NULL;
  g_autofree char *path = NULL;

  g_assert (PLUGIN_IS_GSETTINGS_TEXT_SETTINGS_PROVIDER (self));

  document = foundry_text_settings_provider_dup_document (FOUNDRY_TEXT_SETTINGS_PROVIDER (self));
  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (document));
  buffer = foundry_text_document_dup_buffer (document);
  language_id = foundry_text_buffer_dup_language_id (buffer);

  if (language_id == NULL)
    language_id = g_strdup ("plain-text");

  path = g_strdup_printf ("/app/devsuite/foundry/text/%s/", language_id);
  settings = foundry_settings_new_with_path (context, "app.devsuite.foundry.text", path);

  if (default_settings == NULL)
    default_settings = foundry_settings_new_with_path (context, "app.devsuite.foundry.text", "/app/devsuite/foundry/text/");

  g_set_object (&self->settings, settings);

  g_signal_connect_object (settings,
                           "changed",
                           G_CALLBACK (plugin_gsettings_text_settings_provider_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (default_settings,
                           "changed",
                           G_CALLBACK (plugin_gsettings_text_settings_provider_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  foundry_text_settings_provider_emit_changed (FOUNDRY_TEXT_SETTINGS_PROVIDER (self), 0);
}

static DexFuture *
plugin_gsettings_text_settings_provider_load (FoundryTextSettingsProvider *provider)
{
  PluginGsettingsTextSettingsProvider *self = (PluginGsettingsTextSettingsProvider *)provider;
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryTextBuffer) buffer = NULL;

  g_assert (PLUGIN_IS_GSETTINGS_TEXT_SETTINGS_PROVIDER (self));

  document = foundry_text_settings_provider_dup_document (FOUNDRY_TEXT_SETTINGS_PROVIDER (self));
  buffer = foundry_text_document_dup_buffer (document);

  g_signal_connect_object (buffer,
                           "notify::language-id",
                           G_CALLBACK (plugin_gsettings_text_settings_provider_reload),
                           self,
                           G_CONNECT_SWAPPED);

  plugin_gsettings_text_settings_provider_reload (self);

  return dex_future_new_true ();
}

static DexFuture *
plugin_gsettings_text_settings_provider_unload (FoundryTextSettingsProvider *provider)
{
  PluginGsettingsTextSettingsProvider *self = (PluginGsettingsTextSettingsProvider *)provider;

  g_assert (PLUGIN_IS_GSETTINGS_TEXT_SETTINGS_PROVIDER (self));

  g_clear_object (&self->settings);

  return dex_future_new_true ();
}

static gboolean
plugin_gsettings_text_settings_provider_get_setting (FoundryTextSettingsProvider *provider,
                                                     FoundryTextSetting           setting,
                                                     GValue                      *value)
{
  PluginGsettingsTextSettingsProvider *self = (PluginGsettingsTextSettingsProvider *)provider;
  GEnumClass *klass;
  GEnumValue *enum_value;

  g_assert (PLUGIN_IS_GSETTINGS_TEXT_SETTINGS_PROVIDER (self));
  g_assert (value != NULL);

  if (!(klass = g_type_class_get (FOUNDRY_TYPE_TEXT_SETTING)) ||
      !(enum_value = g_enum_get_value (klass, setting)))
    return FALSE;

  /* First check per-language settings */
  if (self->settings != NULL)
    {
      g_autoptr(GVariant) variant = NULL;

      if ((variant = foundry_settings_get_user_value (self->settings, enum_value->value_nick)))
        return g_settings_get_mapping (value, variant, NULL);
    }

  /* Now fallback to global overrides for all languages */
  if (default_settings != NULL)
    {
      g_autoptr(GVariant) variant = NULL;

      if ((variant = foundry_settings_get_user_value (default_settings, enum_value->value_nick)))
        return g_settings_get_mapping (value, variant, NULL);
    }

  return FALSE;
}

static void
plugin_gsettings_text_settings_provider_class_init (PluginGsettingsTextSettingsProviderClass *klass)
{
  FoundryTextSettingsProviderClass *provider_class = FOUNDRY_TEXT_SETTINGS_PROVIDER_CLASS (klass);

  provider_class->load = plugin_gsettings_text_settings_provider_load;
  provider_class->unload = plugin_gsettings_text_settings_provider_unload;
  provider_class->get_setting = plugin_gsettings_text_settings_provider_get_setting;
}

static void
plugin_gsettings_text_settings_provider_init (PluginGsettingsTextSettingsProvider *self)
{
}
