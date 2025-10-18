/* foundry-markup.h
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

#include <glib-object.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_MARKUP      (foundry_markup_get_type())
#define FOUNDRY_TYPE_MARKUP_KIND (foundry_markup_kind_get_type())

typedef enum _FoundryMarkupKind
{
  FOUNDRY_MARKUP_KIND_PLAINTEXT,
  FOUNDRY_MARKUP_KIND_MARKDOWN,
  FOUNDRY_MARKUP_KIND_HTML,
  FOUNDRY_MARKUP_KIND_PANGO,
} FoundryMarkupKind;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryMarkup, foundry_markup, FOUNDRY, MARKUP, GObject)

FOUNDRY_AVAILABLE_IN_ALL
GType              foundry_markup_kind_get_type (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryMarkup     *foundry_markup_new           (GBytes            *contents,
                                                 FoundryMarkupKind  kind);
FOUNDRY_AVAILABLE_IN_ALL
FoundryMarkup     *foundry_markup_new_plaintext (const char        *message);
FOUNDRY_AVAILABLE_IN_ALL
GBytes            *foundry_markup_dup_contents  (FoundryMarkup     *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryMarkupKind  foundry_markup_get_kind      (FoundryMarkup     *self);
FOUNDRY_AVAILABLE_IN_ALL
char              *foundry_markup_to_string     (FoundryMarkup     *self,
                                                 gsize             *length);

G_END_DECLS
