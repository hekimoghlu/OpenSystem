/* foundry-vcs-reference.h
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

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_VCS_REFERENCE (foundry_vcs_reference_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryVcsReference, foundry_vcs_reference, FOUNDRY, VCS_REFERENCE, GObject)

struct _FoundryVcsReferenceClass
{
  GObjectClass parent_class;

  char      *(*dup_id)      (FoundryVcsReference *self);
  char      *(*dup_title)   (FoundryVcsReference *self);
  gboolean   (*is_symbolic) (FoundryVcsReference *self);
  DexFuture *(*resolve)     (FoundryVcsReference *self);
  DexFuture *(*load_commit) (FoundryVcsReference *self);

  /*< private >*/
  gpointer _reserved[10];
};

FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_reference_dup_id      (FoundryVcsReference *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_reference_dup_title   (FoundryVcsReference *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_vcs_reference_is_symbolic (FoundryVcsReference *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_reference_resolve     (FoundryVcsReference *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_reference_load_commit (FoundryVcsReference *self);

G_END_DECLS
