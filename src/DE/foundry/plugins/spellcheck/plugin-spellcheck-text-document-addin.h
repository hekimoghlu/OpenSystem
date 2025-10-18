/* plugin-spellcheck-text-document-addin.h
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

#include <foundry.h>

G_BEGIN_DECLS

#define PLUGIN_TYPE_SPELLCHECK_TEXT_DOCUMENT_ADDIN (plugin_spellcheck_text_document_addin_get_type())

G_DECLARE_FINAL_TYPE (PluginSpellcheckTextDocumentAddin, plugin_spellcheck_text_document_addin, PLUGIN, SPELLCHECK_TEXT_DOCUMENT_ADDIN, FoundryTextDocumentAddin)

gboolean    plugin_spellcheck_text_document_addin_get_enable_spell_check (PluginSpellcheckTextDocumentAddin *self);
void        plugin_spellcheck_text_document_addin_set_enable_spell_check (PluginSpellcheckTextDocumentAddin *self,
                                                                          gboolean                           enable_spell_check);
char       *plugin_spellcheck_text_document_addin_dup_override_spelling  (PluginSpellcheckTextDocumentAddin *self);
void        plugin_spellcheck_text_document_addin_set_override_spelling  (PluginSpellcheckTextDocumentAddin *self,
                                                                          const char                        *override_spelling);
void        plugin_spellcheck_text_document_addin_update_corrections     (PluginSpellcheckTextDocumentAddin *self);
GMenuModel *plugin_spellcheck_text_document_addin_get_menu               (PluginSpellcheckTextDocumentAddin *self);

G_END_DECLS
