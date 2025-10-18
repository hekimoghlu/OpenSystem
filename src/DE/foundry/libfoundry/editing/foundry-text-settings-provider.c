/* foundry-text-settings-provider.c
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

#include "foundry-text-document.h"
#include "foundry-text-settings-provider-private.h"

typedef struct
{
  GWeakRef        document_wr;
  PeasPluginInfo *plugin_info;
} FoundryTextSettingsProviderPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryTextSettingsProvider, foundry_text_settings_provider, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_DOCUMENT,
  PROP_PLUGIN_INFO,
  N_PROPS
};

enum {
  CHANGED,
  N_SIGNALS
};

static GParamSpec *properties[N_PROPS];
static guint signals[N_SIGNALS];

static void
foundry_text_settings_provider_finalize (GObject *object)
{
  FoundryTextSettingsProvider *self = (FoundryTextSettingsProvider *)object;
  FoundryTextSettingsProviderPrivate *priv = foundry_text_settings_provider_get_instance_private (self);

  g_clear_object (&priv->plugin_info);
  g_weak_ref_clear (&priv->document_wr);

  G_OBJECT_CLASS (foundry_text_settings_provider_parent_class)->finalize (object);
}

static void
foundry_text_settings_provider_get_property (GObject    *object,
                                             guint       prop_id,
                                             GValue     *value,
                                             GParamSpec *pspec)
{
  FoundryTextSettingsProvider *self = FOUNDRY_TEXT_SETTINGS_PROVIDER (object);

  switch (prop_id)
    {
    case PROP_DOCUMENT:
      g_value_take_object (value, foundry_text_settings_provider_dup_document (self));
      break;

    case PROP_PLUGIN_INFO:
      g_value_take_object (value, foundry_text_settings_provider_dup_plugin_info (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_text_settings_provider_set_property (GObject      *object,
                                             guint         prop_id,
                                             const GValue *value,
                                             GParamSpec   *pspec)
{
  FoundryTextSettingsProvider *self = FOUNDRY_TEXT_SETTINGS_PROVIDER (object);
  FoundryTextSettingsProviderPrivate *priv = foundry_text_settings_provider_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_PLUGIN_INFO:
      priv->plugin_info = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_text_settings_provider_class_init (FoundryTextSettingsProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_text_settings_provider_finalize;
  object_class->get_property = foundry_text_settings_provider_get_property;
  object_class->set_property = foundry_text_settings_provider_set_property;

  properties[PROP_DOCUMENT] =
    g_param_spec_object ("document", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_DOCUMENT,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PLUGIN_INFO] =
    g_param_spec_object ("plugin-info", NULL, NULL,
                         PEAS_TYPE_PLUGIN_INFO,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  signals[CHANGED] =
    g_signal_new ("changed",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  G_STRUCT_OFFSET (FoundryTextSettingsProviderClass, changed),
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 1, FOUNDRY_TYPE_TEXT_SETTING);
}

static void
foundry_text_settings_provider_init (FoundryTextSettingsProvider *self)
{
  FoundryTextSettingsProviderPrivate *priv = foundry_text_settings_provider_get_instance_private (self);

  g_weak_ref_init (&priv->document_wr, NULL);
}

DexFuture *
_foundry_text_settings_provider_load (FoundryTextSettingsProvider *self,
                                      FoundryTextDocument         *document)
{
  FoundryTextSettingsProviderPrivate *priv = foundry_text_settings_provider_get_instance_private (self);

  dex_return_error_if_fail (FOUNDRY_IS_TEXT_SETTINGS_PROVIDER (self));
  dex_return_error_if_fail (FOUNDRY_IS_TEXT_DOCUMENT (document));

  g_weak_ref_set (&priv->document_wr, document);

  if (FOUNDRY_TEXT_SETTINGS_PROVIDER_GET_CLASS (self)->load)
    return FOUNDRY_TEXT_SETTINGS_PROVIDER_GET_CLASS (self)->load (self);

  return dex_future_new_true ();
}

DexFuture *
_foundry_text_settings_provider_unload (FoundryTextSettingsProvider *self)
{
  FoundryTextSettingsProviderPrivate *priv = foundry_text_settings_provider_get_instance_private (self);
  DexFuture *ret;

  dex_return_error_if_fail (FOUNDRY_IS_TEXT_SETTINGS_PROVIDER (self));

  if (FOUNDRY_TEXT_SETTINGS_PROVIDER_GET_CLASS (self)->unload)
    ret = FOUNDRY_TEXT_SETTINGS_PROVIDER_GET_CLASS (self)->unload (self);
  else
    ret = dex_future_new_true ();

  g_weak_ref_set (&priv->document_wr, NULL);

  return ret;
}

/**
 * foundry_text_settings_provider_dup_document:
 * @self: a [class@Foundry.TextSettingsProvider]
 *
 * The document the settings should represent, if any.
 *
 * Returns: (transfer full) (nullable):
 */
FoundryTextDocument *
foundry_text_settings_provider_dup_document (FoundryTextSettingsProvider *self)
{
  FoundryTextSettingsProviderPrivate *priv = foundry_text_settings_provider_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS_PROVIDER (self), NULL);

  return g_weak_ref_get (&priv->document_wr);
}

void
foundry_text_settings_provider_emit_changed (FoundryTextSettingsProvider *self,
                                             FoundryTextSetting           setting)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_SETTINGS_PROVIDER (self));
  g_return_if_fail (setting < FOUNDRY_TEXT_SETTING_LAST);

  g_signal_emit (self, signals[CHANGED], 0, setting);
}

gboolean
foundry_text_settings_provider_get_setting (FoundryTextSettingsProvider *self,
                                            FoundryTextSetting           setting,
                                            GValue                      *value)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS_PROVIDER (self), FALSE);
  g_return_val_if_fail (setting > 0, FALSE);
  g_return_val_if_fail (setting < FOUNDRY_TEXT_SETTING_LAST, FALSE);
  g_return_val_if_fail (G_IS_VALUE (value), FALSE);

  if (FOUNDRY_TEXT_SETTINGS_PROVIDER_GET_CLASS (self)->get_setting)
    return FOUNDRY_TEXT_SETTINGS_PROVIDER_GET_CLASS (self)->get_setting (self, setting, value);

  return FALSE;
}

/**
 * foundry_text_settings_provider_dup_plugin_info:
 * @self: a [class@Foundry.TextSettingsProvider]
 *
 * Returns: (transfer full) (nullable):
 */
PeasPluginInfo *
foundry_text_settings_provider_dup_plugin_info (FoundryTextSettingsProvider *self)
{
  FoundryTextSettingsProviderPrivate *priv = foundry_text_settings_provider_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_TEXT_SETTINGS_PROVIDER (self), NULL);

  return priv->plugin_info ? g_object_ref (priv->plugin_info) : NULL;
}

G_DEFINE_ENUM_TYPE (FoundryTextSetting, foundry_text_setting,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_NONE, "none"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_AUTO_INDENT, "auto-indent"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_COMPLETION_AUTO_SELECT, "completion-auto-select"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_COMPLETION_PAGE_SIZE, "completion-page-size"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_CUSTOM_FONT, "custom-font"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_ENABLE_COMPLETION, "enable-completion"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_ENABLE_SNIPPETS, "enable-snippets"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_ENABLE_SPELL_CHECK, "enable-spell-check"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_HIGHLIGHT_CURRENT_LINE, "highlight-current-line"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_HIGHLIGHT_MATCHING_BRACKETS, "highlight-matching-brackets"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_IMPLICIT_TRAILING_NEWLINE, "implicit-trailing-newline"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_INDENT_ON_TAB, "indent-on-tab"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_INDENT_WIDTH, "indent-width"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_INSERT_MATCHING_BRACE, "insert-matching-brace"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_INSERT_SPACES_INSTEAD_OF_TABS, "insert-spaces-instead-of-tabs"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_LINE_HEIGHT, "line-height"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_OVERRIDE_INDENT_WIDTH, "override-indent-width"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_OVERWRITE_MATCHING_BRACE, "overwrite-matching-brace"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_RIGHT_MARGIN_POSITION, "right-margin-position"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_SHOW_DIAGNOSTICS, "show-diagnostics"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES, "show-line-changes"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES_OVERVIEW, "show-line-changes-overview"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_SHOW_LINE_NUMBERS, "show-line-numbers"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_SHOW_RIGHT_MARGIN, "show-right-margin"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_SMART_BACKSPACE, "smart-backspace"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_SMART_HOME_END, "smart-home-end"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_TAB_WIDTH, "tab-width"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_USE_CUSTOM_FONT, "use-custom-font"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_SETTING_WRAP, "wrap"))

G_DEFINE_ENUM_TYPE (FoundryTextWrap, foundry_text_wrap,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_WRAP_NONE, "none"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_WRAP_CHAR, "char"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_WRAP_WORD, "word"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_TEXT_WRAP_WORD_CHAR, "word-char"))
