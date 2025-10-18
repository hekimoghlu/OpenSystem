/* foundry-vcs-branch.h
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

#include "foundry-vcs-reference.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_VCS_BRANCH (foundry_vcs_branch_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryVcsBranch, foundry_vcs_branch, FOUNDRY, VCS_BRANCH, GObject)

struct _FoundryVcsBranchClass
{
  GObjectClass parent_class;

  char      *(*dup_id)      (FoundryVcsBranch *self);
  char      *(*dup_title)   (FoundryVcsBranch *self);
  gboolean   (*is_local)    (FoundryVcsBranch *self);
  DexFuture *(*load_target) (FoundryVcsBranch *self);

  /*< private >*/
  gpointer _reserved[19];
};

FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_vcs_branch_is_local    (FoundryVcsBranch *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_branch_dup_id      (FoundryVcsBranch *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_branch_dup_title   (FoundryVcsBranch *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_branch_load_target (FoundryVcsBranch *self);

G_END_DECLS
