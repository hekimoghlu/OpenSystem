/* foundry-input.h
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

#include <libdex.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_INPUT (foundry_input_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryInput, foundry_input, FOUNDRY, INPUT, GObject)

struct _FoundryInputClass
{
  GObjectClass parent_class;

  /*< private >*/
  gpointer _reserved[15];
};

FOUNDRY_AVAILABLE_IN_ALL
char                  *foundry_input_dup_subtitle  (FoundryInput          *self);
FOUNDRY_AVAILABLE_IN_ALL
char                  *foundry_input_dup_title     (FoundryInput          *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryInputValidator *foundry_input_dup_validator (FoundryInput          *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture             *foundry_input_validate      (FoundryInput          *self);

G_END_DECLS
