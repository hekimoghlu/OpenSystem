/* foundry-text-settings-provider.h
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

#pragma once

#include <libpeas.h>

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEXT_SETTINGS_PROVIDER (foundry_text_settings_provider_get_type())
#define FOUNDRY_TYPE_TEXT_SETTING           (foundry_text_setting_get_type())

typedef enum _FoundryTextSetting
{
  FOUNDRY_TEXT_SETTING_NONE = 0,
  FOUNDRY_TEXT_SETTING_AUTO_INDENT,
  FOUNDRY_TEXT_SETTING_COMPLETION_AUTO_SELECT,
  FOUNDRY_TEXT_SETTING_COMPLETION_PAGE_SIZE,
  FOUNDRY_TEXT_SETTING_CUSTOM_FONT,
  FOUNDRY_TEXT_SETTING_ENABLE_COMPLETION,
  FOUNDRY_TEXT_SETTING_ENABLE_SNIPPETS,
  FOUNDRY_TEXT_SETTING_ENABLE_SPELL_CHECK,
  FOUNDRY_TEXT_SETTING_HIGHLIGHT_CURRENT_LINE,
  FOUNDRY_TEXT_SETTING_HIGHLIGHT_MATCHING_BRACKETS,
  FOUNDRY_TEXT_SETTING_IMPLICIT_TRAILING_NEWLINE,
  FOUNDRY_TEXT_SETTING_INDENT_ON_TAB,
  FOUNDRY_TEXT_SETTING_INDENT_WIDTH,
  FOUNDRY_TEXT_SETTING_INSERT_MATCHING_BRACE,
  FOUNDRY_TEXT_SETTING_INSERT_SPACES_INSTEAD_OF_TABS,
  FOUNDRY_TEXT_SETTING_LINE_HEIGHT,
  FOUNDRY_TEXT_SETTING_OVERRIDE_INDENT_WIDTH,
  FOUNDRY_TEXT_SETTING_OVERWRITE_MATCHING_BRACE,
  FOUNDRY_TEXT_SETTING_RIGHT_MARGIN_POSITION,
  FOUNDRY_TEXT_SETTING_SHOW_DIAGNOSTICS,
  FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES,
  FOUNDRY_TEXT_SETTING_SHOW_LINE_CHANGES_OVERVIEW,
  FOUNDRY_TEXT_SETTING_SHOW_LINE_NUMBERS,
  FOUNDRY_TEXT_SETTING_SHOW_RIGHT_MARGIN,
  FOUNDRY_TEXT_SETTING_SMART_BACKSPACE,
  FOUNDRY_TEXT_SETTING_SMART_HOME_END,
  FOUNDRY_TEXT_SETTING_TAB_WIDTH,
  FOUNDRY_TEXT_SETTING_USE_CUSTOM_FONT,
  FOUNDRY_TEXT_SETTING_WRAP,
} FoundryTextSetting;

/* Not part of ABI */
#define FOUNDRY_TEXT_SETTING_LAST (FOUNDRY_TEXT_SETTING_WRAP+1)

typedef enum _FoundryTextWrap
{
  FOUNDRY_TEXT_WRAP_NONE = 0,
  FOUNDRY_TEXT_WRAP_CHAR = 1,
  FOUNDRY_TEXT_WRAP_WORD = 2,
  FOUNDRY_TEXT_WRAP_WORD_CHAR = 3,
} FoundryTextWrap;

#define FOUNDRY_TYPE_TEXT_WRAP (foundry_text_wrap_get_type())

FOUNDRY_AVAILABLE_IN_ALL
GType foundry_text_wrap_get_type (void) G_GNUC_CONST;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryTextSettingsProvider, foundry_text_settings_provider, FOUNDRY, TEXT_SETTINGS_PROVIDER, FoundryContextual)

struct _FoundryTextSettingsProviderClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*load)        (FoundryTextSettingsProvider *self);
  DexFuture *(*unload)      (FoundryTextSettingsProvider *self);
  void       (*changed)     (FoundryTextSettingsProvider *self,
                             FoundryTextSetting           setting);
  gboolean   (*get_setting) (FoundryTextSettingsProvider *self,
                             FoundryTextSetting           setting,
                             GValue                      *value);

  /*< private >*/
  gpointer _reserved[4];
};

FOUNDRY_AVAILABLE_IN_ALL
GType                foundry_text_setting_get_type                  (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
PeasPluginInfo      *foundry_text_settings_provider_dup_plugin_info (FoundryTextSettingsProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTextDocument *foundry_text_settings_provider_dup_document    (FoundryTextSettingsProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_text_settings_provider_emit_changed    (FoundryTextSettingsProvider *self,
                                                                    FoundryTextSetting           setting);
FOUNDRY_AVAILABLE_IN_ALL
gboolean             foundry_text_settings_provider_get_setting     (FoundryTextSettingsProvider *self,
                                                                     FoundryTextSetting           setting,
                                                                     GValue                      *value);

G_END_DECLS

