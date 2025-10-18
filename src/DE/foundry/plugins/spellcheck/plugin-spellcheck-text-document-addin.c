/* plugin-spellcheck-text-document-addin.c
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

#include <libspelling.h>

#include <foundry-gtk.h>

#include "plugin-spellcheck-text-document-addin.h"

#define METADATA_SPELLING "metadata::foundry-spelling-language"

struct _PluginSpellcheckTextDocumentAddin
{
  FoundryTextDocumentAddin   parent_instance;
  SpellingTextBufferAdapter *adapter;
  FoundryTextSettings       *settings;
  char                      *override_spelling;
  guint                      enable_spell_check : 1;
};

enum {
  PROP_0,
  PROP_ENABLE_SPELL_CHECK,
  PROP_OVERRIDE_SPELLING,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (PluginSpellcheckTextDocumentAddin, plugin_spellcheck_text_document_addin, FOUNDRY_TYPE_TEXT_DOCUMENT_ADDIN)

static GParamSpec *properties[N_PROPS];

static DexFuture *
plugin_spellcheck_text_document_addin_post_save (FoundryTextDocumentAddin *addin)
{
  PluginSpellcheckTextDocumentAddin *self = (PluginSpellcheckTextDocumentAddin *)addin;
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GFileInfo) info = NULL;
  g_autoptr(GFile) file = NULL;

  g_assert (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  file_manager = foundry_context_dup_file_manager (context);
  document = foundry_text_document_addin_dup_document (addin);
  file = foundry_text_document_dup_file (document);

  info = g_file_info_new ();

  if (self->override_spelling != NULL)
    g_file_info_set_attribute_string (info, METADATA_SPELLING, self->override_spelling);

  return foundry_file_manager_write_metadata (file_manager, file, info);
}

static DexFuture *
plugin_spellcheck_text_document_addin_pre_load (FoundryTextDocumentAddin *addin)
{
  PluginSpellcheckTextDocumentAddin *self = (PluginSpellcheckTextDocumentAddin *)addin;

  g_assert (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (addin));

  g_clear_object (&self->adapter);

  return dex_future_new_true ();
}

static DexFuture *
plugin_spellcheck_text_document_addin_post_load_fiber (gpointer user_data)
{
  PluginSpellcheckTextDocumentAddin *self = user_data;
  g_autoptr(FoundryTextDocumentAddin) addin = NULL;
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryTextSettings) settings = NULL;
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GFileInfo) info = NULL;
  g_autoptr(GFile) file = NULL;

  g_assert (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  file_manager = foundry_context_dup_file_manager (context);
  document = foundry_text_document_addin_dup_document (FOUNDRY_TEXT_DOCUMENT_ADDIN (self));
  file = foundry_text_document_dup_file (document);

  if ((info = dex_await_object (foundry_file_manager_read_metadata (file_manager, file, METADATA_SPELLING), NULL)))
    {
      const char *str = g_file_info_get_attribute_string (info, METADATA_SPELLING);

      if (str != NULL)
        plugin_spellcheck_text_document_addin_set_override_spelling (self, str);
    }

  if ((settings = dex_await_object (foundry_text_document_load_settings (document), NULL)))
    g_object_bind_property (settings, "enable-spell-check", self, "enable-spell-check",
                            G_BINDING_SYNC_CREATE);

  if (self->enable_spell_check)
    spelling_text_buffer_adapter_set_enabled (self->adapter, self->enable_spell_check);

  return dex_future_new_true ();
}

static DexFuture *
plugin_spellcheck_text_document_addin_post_load (FoundryTextDocumentAddin *addin)
{
  PluginSpellcheckTextDocumentAddin *self = (PluginSpellcheckTextDocumentAddin *)addin;
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryTextBuffer) buffer = NULL;

  g_assert (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self));

  document = foundry_text_document_addin_dup_document (addin);
  buffer = foundry_text_document_dup_buffer (document);

  if (!FOUNDRY_IS_SOURCE_BUFFER (buffer))
    return foundry_future_new_not_supported ();

  self->adapter = spelling_text_buffer_adapter_new (GTK_SOURCE_BUFFER (buffer),
                                                    spelling_checker_get_default ());

  spelling_text_buffer_adapter_set_enabled (self->adapter, FALSE);

  if (self->override_spelling)
    spelling_text_buffer_adapter_set_language (self->adapter, self->override_spelling);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_spellcheck_text_document_addin_post_load_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
plugin_spellcheck_text_document_addin_unload (FoundryTextDocumentAddin *addin)
{
  PluginSpellcheckTextDocumentAddin *self = (PluginSpellcheckTextDocumentAddin *)addin;

  g_assert (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self));

  g_clear_object (&self->adapter);
  g_clear_object (&self->settings);

  return dex_future_new_true ();
}

static void
plugin_spellcheck_text_document_addin_finalize (GObject *object)
{
  PluginSpellcheckTextDocumentAddin *self = (PluginSpellcheckTextDocumentAddin *)object;

  g_clear_pointer (&self->override_spelling, g_free);
  g_clear_object (&self->settings);

  G_OBJECT_CLASS (plugin_spellcheck_text_document_addin_parent_class)->finalize (object);
}

static void
plugin_spellcheck_text_document_addin_get_property (GObject    *object,
                                                    guint       prop_id,
                                                    GValue     *value,
                                                    GParamSpec *pspec)
{
  PluginSpellcheckTextDocumentAddin *self = PLUGIN_SPELLCHECK_TEXT_DOCUMENT_ADDIN (object);

  switch (prop_id)
    {
    case PROP_ENABLE_SPELL_CHECK:
      g_value_set_boolean (value, plugin_spellcheck_text_document_addin_get_enable_spell_check (self));
      break;

    case PROP_OVERRIDE_SPELLING:
      g_value_take_string (value, plugin_spellcheck_text_document_addin_dup_override_spelling (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_spellcheck_text_document_addin_set_property (GObject      *object,
                                                    guint         prop_id,
                                                    const GValue *value,
                                                    GParamSpec   *pspec)
{
  PluginSpellcheckTextDocumentAddin *self = PLUGIN_SPELLCHECK_TEXT_DOCUMENT_ADDIN (object);

  switch (prop_id)
    {
    case PROP_ENABLE_SPELL_CHECK:
      plugin_spellcheck_text_document_addin_set_enable_spell_check (self, g_value_get_boolean (value));
      break;

    case PROP_OVERRIDE_SPELLING:
      plugin_spellcheck_text_document_addin_set_override_spelling (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_spellcheck_text_document_addin_class_init (PluginSpellcheckTextDocumentAddinClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryTextDocumentAddinClass *addin_class = FOUNDRY_TEXT_DOCUMENT_ADDIN_CLASS (klass);

  object_class->finalize = plugin_spellcheck_text_document_addin_finalize;
  object_class->get_property = plugin_spellcheck_text_document_addin_get_property;
  object_class->set_property = plugin_spellcheck_text_document_addin_set_property;

  addin_class->pre_load = plugin_spellcheck_text_document_addin_pre_load;
  addin_class->post_load = plugin_spellcheck_text_document_addin_post_load;
  addin_class->post_save = plugin_spellcheck_text_document_addin_post_save;
  addin_class->unload = plugin_spellcheck_text_document_addin_unload;

  properties[PROP_ENABLE_SPELL_CHECK] =
    g_param_spec_boolean ("enable-spell-check", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_OVERRIDE_SPELLING] =
    g_param_spec_string ("override-spelling", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_spellcheck_text_document_addin_init (PluginSpellcheckTextDocumentAddin *self)
{
  self->enable_spell_check = TRUE;
}

char *
plugin_spellcheck_text_document_addin_dup_override_spelling (PluginSpellcheckTextDocumentAddin *self)
{
  g_return_val_if_fail (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self), NULL);

  return g_strdup (self->override_spelling);
}

void
plugin_spellcheck_text_document_addin_set_override_spelling (PluginSpellcheckTextDocumentAddin *self,
                                                             const char                        *override_spelling)
{
  g_return_if_fail (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self));

  if (g_set_str (&self->override_spelling, override_spelling))
    {
      if (self->adapter != NULL)
        {
          if (override_spelling == NULL)
            {
              SpellingProvider *provider = spelling_provider_get_default ();

              if (provider != NULL)
                override_spelling = spelling_provider_get_default_code (provider);
            }

          if (override_spelling != NULL)
            spelling_text_buffer_adapter_set_language (self->adapter, override_spelling);
        }

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_OVERRIDE_SPELLING]);
    }
}

gboolean
plugin_spellcheck_text_document_addin_get_enable_spell_check (PluginSpellcheckTextDocumentAddin *self)
{
  g_autoptr(FoundryTextDocument) document = NULL;
  g_autoptr(FoundryTextBuffer) buffer = NULL;

  g_return_val_if_fail (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self), FALSE);

  document = foundry_text_document_addin_dup_document (FOUNDRY_TEXT_DOCUMENT_ADDIN (self));
  buffer = foundry_text_document_dup_buffer (document);

  if (!GTK_SOURCE_IS_BUFFER (buffer))
    return FALSE;

  if (gtk_source_buffer_get_loading (GTK_SOURCE_BUFFER (buffer)))
    return FALSE;

  return self->enable_spell_check;
}

void
plugin_spellcheck_text_document_addin_set_enable_spell_check (PluginSpellcheckTextDocumentAddin *self,
                                                             gboolean                           enable_spell_check)
{
  g_return_if_fail (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self));

  enable_spell_check = !!enable_spell_check;

  if (self->enable_spell_check != enable_spell_check)
    {
      self->enable_spell_check = enable_spell_check;

      if (self->adapter != NULL)
        {
          spelling_text_buffer_adapter_set_enabled (self->adapter,
                                                    plugin_spellcheck_text_document_addin_get_enable_spell_check (self));
          spelling_text_buffer_adapter_invalidate_all (self->adapter);
        }

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ENABLE_SPELL_CHECK]);
    }
}

void
plugin_spellcheck_text_document_addin_update_corrections (PluginSpellcheckTextDocumentAddin *self)
{
  g_return_if_fail (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self));

  if (self->adapter != NULL)
    spelling_text_buffer_adapter_update_corrections (self->adapter);
}

GMenuModel *
plugin_spellcheck_text_document_addin_get_menu (PluginSpellcheckTextDocumentAddin *self)
{
  g_return_val_if_fail (PLUGIN_IS_SPELLCHECK_TEXT_DOCUMENT_ADDIN (self), NULL);

  if (self->adapter != NULL)
    return spelling_text_buffer_adapter_get_menu_model (self->adapter);

  return NULL;
}
