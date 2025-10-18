/* foundry-on-type-formatter.h
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

#include <glib-object.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_ON_TYPE_FORMATTER (foundry_on_type_formatter_get_type())
#define FOUNDRY_TYPE_MODIFIER_TYPE  (foundry_modifier_type_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryOnTypeFormatter, foundry_on_type_formatter, FOUNDRY, ON_TYPE_FORMATTER, GObject)

typedef enum _FoundryModifierType
{
  FOUNDRY_MODIFIER_CONTROL = 1 << 0,
  FOUNDRY_MODIFIER_SHIFT   = 1 << 1,
  FOUNDRY_MODIFIER_ALT     = 1 << 2,
  FOUNDRY_MODIFIER_SUPER   = 1 << 3,
  FOUNDRY_MODIFIER_COMMAND = 1 << 4,
} FoundryModifierType;

struct _FoundryOnTypeFormatterClass
{
  GObjectClass parent_class;

  gboolean (*is_trigger) (FoundryOnTypeFormatter *self,
                          FoundryTextDocument    *document,
                          const FoundryTextIter  *iter,
                          FoundryModifierType     state,
                          guint                   keyval);
  void     (*indent)     (FoundryOnTypeFormatter *self,
                          FoundryTextDocument    *document,
                          FoundryTextIter        *iter);

  /*< private >*/
  gpointer _reserved[13];
};

FOUNDRY_AVAILABLE_IN_ALL
GType    foundry_modifier_type_get_type       (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
gboolean foundry_on_type_formatter_is_trigger (FoundryOnTypeFormatter *self,
                                               FoundryTextDocument    *document,
                                               const FoundryTextIter  *iter,
                                               FoundryModifierType     state,
                                               guint                   keyval);
FOUNDRY_AVAILABLE_IN_ALL
void     foundry_on_type_formatter_indent     (FoundryOnTypeFormatter *self,
                                               FoundryTextDocument    *document,
                                               FoundryTextIter        *iter);

G_END_DECLS
