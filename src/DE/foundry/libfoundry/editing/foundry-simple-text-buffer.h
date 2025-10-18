/* foundry-simple-text-buffer.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-text-buffer.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SIMPLE_TEXT_BUFFER (foundry_simple_text_buffer_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundrySimpleTextBuffer, foundry_simple_text_buffer, FOUNDRY, SIMPLE_TEXT_BUFFER, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryTextBuffer *foundry_simple_text_buffer_new             (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryTextBuffer *foundry_simple_text_buffer_new_for_string  (const char              *string,
                                                               gssize                   len);
FOUNDRY_AVAILABLE_IN_ALL
void               foundry_simple_text_buffer_set_language_id (FoundrySimpleTextBuffer *self,
                                                               const char              *language_id);
FOUNDRY_AVAILABLE_IN_ALL
void               foundry_simple_text_buffer_set_text        (FoundrySimpleTextBuffer *self,
                                                               const char              *text,
                                                               gssize                   text_len);

G_END_DECLS
