/* foundry-text-settings.h
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

#include "foundry-contextual.h"
#include "foundry-text-settings-provider.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEXT_SETTINGS (foundry_text_settings_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryTextSettings, foundry_text_settings, FOUNDRY, TEXT_SETTINGS, FoundryContextual)

FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_auto_indent                   (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_auto_indent                   (FoundryTextSettings *self,
                                                                          gboolean             auto_indent);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_completion_auto_select        (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_completion_auto_select        (FoundryTextSettings *self,
                                                                          gboolean             completion_auto_select);
FOUNDRY_AVAILABLE_IN_ALL
guint            foundry_text_settings_get_completion_page_size          (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_completion_page_size          (FoundryTextSettings *self,
                                                                          guint                completion_page_size);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_enable_completion             (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_enable_completion             (FoundryTextSettings *self,
                                                                          gboolean             enable_completion);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_enable_snippets               (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_enable_snippets               (FoundryTextSettings *self,
                                                                          gboolean             enable_snippets);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_enable_spell_check            (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_enable_spell_check            (FoundryTextSettings *self,
                                                                          gboolean             enable_spell_check);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_highlight_current_line        (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_highlight_current_line        (FoundryTextSettings *self,
                                                                          gboolean             highlight_current_line);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_highlight_matching_brackets   (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_highlight_matching_brackets   (FoundryTextSettings *self,
                                                                          gboolean             highlight_matching_brackets);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_implicit_trailing_newline     (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_implicit_trailing_newline     (FoundryTextSettings *self,
                                                                          gboolean             implicit_trailing_newline);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_indent_on_tab                 (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_indent_on_tab                 (FoundryTextSettings *self,
                                                                          gboolean             indent_on_tab);
FOUNDRY_AVAILABLE_IN_ALL
guint            foundry_text_settings_get_indent_width                  (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_indent_width                  (FoundryTextSettings *self,
                                                                          guint                indent_width);
FOUNDRY_AVAILABLE_IN_ALL
double           foundry_text_settings_get_line_height                   (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_line_height                   (FoundryTextSettings *self,
                                                                          double               line_height);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_insert_matching_brace         (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_insert_matching_brace         (FoundryTextSettings *self,
                                                                          gboolean             insert_matching_brace);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_insert_spaces_instead_of_tabs (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_insert_spaces_instead_of_tabs (FoundryTextSettings *self,
                                                                          gboolean             insert_spaces_instead_of_tabs);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_override_indent_width         (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_override_indent_width         (FoundryTextSettings *self,
                                                                          gboolean             override_indent_width);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_overwrite_matching_brace      (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_overwrite_matching_brace      (FoundryTextSettings *self,
                                                                          gboolean             overwrite_matching_brace);
FOUNDRY_AVAILABLE_IN_ALL
guint            foundry_text_settings_get_right_margin_position         (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_right_margin_position         (FoundryTextSettings *self,
                                                                          guint                right_margin_position);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_show_diagnostics              (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_show_diagnostics              (FoundryTextSettings *self,
                                                                          gboolean             show_diagnostics);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_show_line_changes             (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_show_line_changes             (FoundryTextSettings *self,
                                                                          gboolean             show_line_changes);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_show_line_changes_overview    (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_show_line_changes_overview    (FoundryTextSettings *self,
                                                                          gboolean             show_line_changes_overview);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_show_line_numbers             (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_show_line_numbers             (FoundryTextSettings *self,
                                                                          gboolean             show_line_numbers);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_show_right_margin             (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_show_right_margin             (FoundryTextSettings *self,
                                                                          gboolean             show_right_margin);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_smart_backspace               (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_smart_backspace               (FoundryTextSettings *self,
                                                                          gboolean             smart_backspace);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_smart_home_end                (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_smart_home_end                (FoundryTextSettings *self,
                                                                          gboolean             smart_home_end);
FOUNDRY_AVAILABLE_IN_ALL
guint            foundry_text_settings_get_tab_width                     (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_tab_width                     (FoundryTextSettings *self,
                                                                          guint                tab_width);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_text_settings_get_use_custom_font               (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_use_custom_font               (FoundryTextSettings *self,
                                                                          gboolean             use_custom_font);
FOUNDRY_AVAILABLE_IN_ALL
char            *foundry_text_settings_dup_custom_font                   (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_custom_font                   (FoundryTextSettings *self,
                                                                          const char          *custom_font);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTextWrap  foundry_text_settings_get_wrap                          (FoundryTextSettings *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_settings_set_wrap                          (FoundryTextSettings *self,
                                                                          FoundryTextWrap      wrap);

G_END_DECLS
