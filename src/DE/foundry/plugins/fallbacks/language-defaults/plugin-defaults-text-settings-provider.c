/* plugin-defaults-text-settings-provider.c
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

#define G_SETTINGS_ENABLE_BACKEND
#include <gio/gsettingsbackend.h>

#include "plugin-defaults-text-settings-provider.h"

#include "gsettings-mapping.h"

struct _PluginDefaultsTextSettingsProvider
{
  FoundryTextSettingsProvider  parent_instance;
  GSettings                   *settings;
};

G_DEFINE_FINAL_TYPE (PluginDefaultsTextSettingsProvider, plugin_defaults_text_settings_provider, FOUNDRY_TYPE_TEXT_SETTINGS_PROVIDER)

static void
plugin_defaults_text_settings_provider_reload (PluginDefaultsTextSettingsProvider *self)
{
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryTextBuffer) buffer = NULL;
  g_autofree char *language_id = NULL;

  g_assert (PLUGIN_IS_DEFAULTS_TEXT_SETTINGS_PROVIDER (self));

  g_clear_object (&self->settings);

  document = foundry_text_settings_provider_dup_document (FOUNDRY_TEXT_SETTINGS_PROVIDER (self));
  buffer = foundry_text_document_dup_buffer (document);
  language_id = foundry_text_buffer_dup_language_id (buffer);

  if (language_id != NULL)
    {
      g_autoptr(GSettingsBackend) backend = NULL;
      g_autoptr(GSettings) settings = NULL;
      g_autofree char *path = NULL;

      backend = g_keyfile_settings_backend_new (PACKAGE_DATADIR "/language-defaults",
                                                "/app/devsuite/foundry/text/",
                                                "app.devsuite.foundry.text");
      path = g_strdup_printf ("/app/devsuite/foundry/text/%s/", language_id);
      settings = g_settings_new_with_backend_and_path ("app.devsuite.foundry.text", backend, path);

      self->settings = g_steal_pointer (&settings);
    }

  foundry_text_settings_provider_emit_changed (FOUNDRY_TEXT_SETTINGS_PROVIDER (self), 0);
}

static DexFuture *
plugin_defaults_text_settings_provider_load (FoundryTextSettingsProvider *provider)
{
  PluginDefaultsTextSettingsProvider *self = (PluginDefaultsTextSettingsProvider *)provider;
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryTextBuffer) buffer = NULL;

  g_assert (PLUGIN_IS_DEFAULTS_TEXT_SETTINGS_PROVIDER (self));

  document = foundry_text_settings_provider_dup_document (FOUNDRY_TEXT_SETTINGS_PROVIDER (self));
  buffer = foundry_text_document_dup_buffer (document);

  g_signal_connect_object (buffer,
                           "notify::language-id",
                           G_CALLBACK (plugin_defaults_text_settings_provider_reload),
                           self,
                           G_CONNECT_SWAPPED);

  plugin_defaults_text_settings_provider_reload (self);

  return dex_future_new_true ();
}

static DexFuture *
plugin_defaults_text_settings_provider_unload (FoundryTextSettingsProvider *provider)
{
  PluginDefaultsTextSettingsProvider *self = (PluginDefaultsTextSettingsProvider *)provider;

  g_assert (PLUGIN_IS_DEFAULTS_TEXT_SETTINGS_PROVIDER (self));

  return dex_future_new_true ();
}

static gboolean
plugin_defaults_text_settings_provider_get_setting (FoundryTextSettingsProvider *provider,
                                                    FoundryTextSetting           setting,
                                                    GValue                      *value)
{
  PluginDefaultsTextSettingsProvider *self = (PluginDefaultsTextSettingsProvider *)provider;
  g_autoptr(GVariant) variant = NULL;
  GEnumClass *klass;
  GEnumValue *enum_value;

  g_assert (PLUGIN_IS_DEFAULTS_TEXT_SETTINGS_PROVIDER (self));
  g_assert (value != NULL);

  if (self->settings == NULL)
    return FALSE;

  klass = g_type_class_get (FOUNDRY_TYPE_TEXT_SETTING);

  if (!(enum_value = g_enum_get_value (klass, setting)))
    return FALSE;

  if (!(variant = g_settings_get_user_value (self->settings, enum_value->value_nick)))
    return FALSE;

  return g_settings_get_mapping (value, variant, NULL);
}

static void
plugin_defaults_text_settings_provider_class_init (PluginDefaultsTextSettingsProviderClass *klass)
{
  FoundryTextSettingsProviderClass *provider_class = FOUNDRY_TEXT_SETTINGS_PROVIDER_CLASS (klass);

  provider_class->load = plugin_defaults_text_settings_provider_load;
  provider_class->unload = plugin_defaults_text_settings_provider_unload;
  provider_class->get_setting = plugin_defaults_text_settings_provider_get_setting;
}

static void
plugin_defaults_text_settings_provider_init (PluginDefaultsTextSettingsProvider *self)
{
}
