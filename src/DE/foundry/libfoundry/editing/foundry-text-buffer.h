/* foundry-text-buffer.h
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

#include <libdex.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEXT_BUFFER (foundry_text_buffer_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_INTERFACE (FoundryTextBuffer, foundry_text_buffer, FOUNDRY, TEXT_BUFFER, GObject)

struct _FoundryTextBufferInterface
{
  GTypeInterface parent_iface;

  GBytes    *(*dup_contents)      (FoundryTextBuffer  *self);
  char      *(*dup_language_id)   (FoundryTextBuffer  *self);
  DexFuture *(*settle)            (FoundryTextBuffer  *self);
  gboolean   (*apply_edit)        (FoundryTextBuffer  *self,
                                   FoundryTextEdit    *edit);
  void       (*iter_init)         (FoundryTextBuffer  *self,
                                   FoundryTextIter    *iter);
  gint64     (*get_change_count)  (FoundryTextBuffer  *self);
};

FOUNDRY_AVAILABLE_IN_ALL
GBytes    *foundry_text_buffer_dup_contents     (FoundryTextBuffer  *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_text_buffer_dup_language_id  (FoundryTextBuffer *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_text_buffer_settle           (FoundryTextBuffer  *self) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_text_buffer_apply_edit       (FoundryTextBuffer  *self,
                                                 FoundryTextEdit    *edit);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_text_buffer_get_start_iter   (FoundryTextBuffer *self,
                                                 FoundryTextIter   *iter);
FOUNDRY_AVAILABLE_IN_ALL
void       foundry_text_buffer_emit_changed     (FoundryTextBuffer *self);
FOUNDRY_AVAILABLE_IN_ALL
gint64     foundry_text_buffer_get_change_count (FoundryTextBuffer *self);

G_END_DECLS
