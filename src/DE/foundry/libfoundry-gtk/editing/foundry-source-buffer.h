/* foundry-source-buffer.h
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
#include <gtksourceview/gtksource.h>

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SOURCE_BUFFER (foundry_source_buffer_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundrySourceBuffer, foundry_source_buffer, FOUNDRY, SOURCE_BUFFER, GtkSourceBuffer)

FOUNDRY_AVAILABLE_IN_ALL
FoundryContext      *foundry_source_buffer_dup_context           (FoundrySourceBuffer *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean             foundry_source_buffer_get_enable_spellcheck (FoundrySourceBuffer *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_source_buffer_set_enable_spellcheck (FoundrySourceBuffer *self,
                                                                  gboolean             enable_spellcheck);
FOUNDRY_AVAILABLE_IN_ALL
char                *foundry_source_buffer_dup_override_spelling (FoundrySourceBuffer *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_source_buffer_set_override_spelling (FoundrySourceBuffer *self,
                                                                  const char          *override_spelling);
FOUNDRY_AVAILABLE_IN_ALL
char                *foundry_source_buffer_dup_override_syntax   (FoundrySourceBuffer *self);
FOUNDRY_AVAILABLE_IN_ALL
void                 foundry_source_buffer_set_override_syntax   (FoundrySourceBuffer *self,
                                                                  const char          *override_syntax);

G_END_DECLS
