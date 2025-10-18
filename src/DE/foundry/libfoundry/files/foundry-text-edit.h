/* foundry-text-edit.h
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

#include <gio/gio.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEXT_EDIT (foundry_text_edit_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryTextEdit, foundry_text_edit, FOUNDRY, TEXT_EDIT, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryTextEdit *foundry_text_edit_new             (GFile                 *file,
                                                    guint                  begin_line,
                                                    int                    begin_line_offset,
                                                    guint                  end_line,
                                                    int                    end_line_offset,
                                                    const char            *replacement);
FOUNDRY_AVAILABLE_IN_ALL
GFile           *foundry_text_edit_dup_file        (FoundryTextEdit       *self);
FOUNDRY_AVAILABLE_IN_ALL
char            *foundry_text_edit_dup_replacement (FoundryTextEdit       *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_text_edit_get_range       (FoundryTextEdit       *self,
                                                    guint                 *begin_line,
                                                    int                   *begin_line_offset,
                                                    guint                 *end_line,
                                                    int                   *end_line_offset);
FOUNDRY_AVAILABLE_IN_ALL
int              foundry_text_edit_compare         (const FoundryTextEdit *a,
                                                    const FoundryTextEdit *b);

G_END_DECLS
