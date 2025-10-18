/*
 * spelling-text-buffer-adapter.h
 *
 * Copyright 2021-2023 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <gtksourceview/gtksource.h>

#include "spelling-types.h"
#include "spelling-version-macros.h"

G_BEGIN_DECLS

#define SPELLING_TYPE_TEXT_BUFFER_ADAPTER (spelling_text_buffer_adapter_get_type())

SPELLING_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (SpellingTextBufferAdapter, spelling_text_buffer_adapter, SPELLING, TEXT_BUFFER_ADAPTER, GObject)

SPELLING_AVAILABLE_IN_ALL
SpellingTextBufferAdapter *spelling_text_buffer_adapter_new                (GtkSourceBuffer           *buffer,
                                                                            SpellingChecker           *checker);
SPELLING_AVAILABLE_IN_ALL
GtkSourceBuffer           *spelling_text_buffer_adapter_get_buffer         (SpellingTextBufferAdapter *self);
SPELLING_AVAILABLE_IN_ALL
gboolean                   spelling_text_buffer_adapter_get_enabled        (SpellingTextBufferAdapter *self);
SPELLING_AVAILABLE_IN_ALL
void                       spelling_text_buffer_adapter_set_enabled        (SpellingTextBufferAdapter *self,
                                                                            gboolean                   enabled);
SPELLING_AVAILABLE_IN_ALL
SpellingChecker           *spelling_text_buffer_adapter_get_checker        (SpellingTextBufferAdapter *self);
SPELLING_AVAILABLE_IN_ALL
void                       spelling_text_buffer_adapter_set_checker        (SpellingTextBufferAdapter *self,
                                                                            SpellingChecker           *checker);
SPELLING_AVAILABLE_IN_ALL
const char                *spelling_text_buffer_adapter_get_language       (SpellingTextBufferAdapter *self);
SPELLING_AVAILABLE_IN_ALL
void                       spelling_text_buffer_adapter_set_language       (SpellingTextBufferAdapter *self,
                                                                            const char                *language);
SPELLING_AVAILABLE_IN_ALL
void                       spelling_text_buffer_adapter_invalidate_all     (SpellingTextBufferAdapter *self);
SPELLING_AVAILABLE_IN_ALL
GtkTextTag                *spelling_text_buffer_adapter_get_tag            (SpellingTextBufferAdapter *self);
SPELLING_AVAILABLE_IN_ALL
GMenuModel                *spelling_text_buffer_adapter_get_menu_model     (SpellingTextBufferAdapter *self);
SPELLING_AVAILABLE_IN_ALL
void                       spelling_text_buffer_adapter_update_corrections (SpellingTextBufferAdapter *self);

G_END_DECLS
