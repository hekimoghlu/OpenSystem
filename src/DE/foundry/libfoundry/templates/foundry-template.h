/* foundry-template.h
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

#define FOUNDRY_TYPE_TEMPLATE (foundry_template_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryTemplate, foundry_template, FOUNDRY, TEMPLATE, GObject)

struct _FoundryTemplateClass
{
  GObjectClass parent_class;

  char          *(*dup_id)          (FoundryTemplate *self);
  char          *(*dup_description) (FoundryTemplate *self);
  char         **(*dup_tags)        (FoundryTemplate *self);
  FoundryInput  *(*dup_input)       (FoundryTemplate *self);
  DexFuture     *(*expand)          (FoundryTemplate *self);

  /*< private >*/
  gpointer _reserved[10];
};

FOUNDRY_AVAILABLE_IN_ALL
char          *foundry_template_dup_id          (FoundryTemplate *self);
FOUNDRY_AVAILABLE_IN_ALL
char         **foundry_template_dup_tags        (FoundryTemplate *self);
FOUNDRY_AVAILABLE_IN_ALL
char          *foundry_template_dup_description (FoundryTemplate *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryInput  *foundry_template_dup_input       (FoundryTemplate *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture     *foundry_template_expand          (FoundryTemplate *self);

G_END_DECLS
