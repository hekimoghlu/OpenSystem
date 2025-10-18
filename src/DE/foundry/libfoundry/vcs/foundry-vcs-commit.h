/* foundry-vcs-commit.h
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

#define FOUNDRY_TYPE_VCS_COMMIT (foundry_vcs_commit_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryVcsCommit, foundry_vcs_commit, FOUNDRY, VCS_COMMIT, GObject)

struct _FoundryVcsCommitClass
{
  GObjectClass parent_class;

  char                *(*dup_id)        (FoundryVcsCommit *self);
  char                *(*dup_title)     (FoundryVcsCommit *self);
  FoundryVcsSignature *(*dup_author)    (FoundryVcsCommit *self);
  FoundryVcsSignature *(*dup_committer) (FoundryVcsCommit *self);
  guint                (*get_n_parents) (FoundryVcsCommit *self);
  DexFuture           *(*load_parent)   (FoundryVcsCommit *self,
                                         guint             index);
  DexFuture           *(*load_tree)     (FoundryVcsCommit *self);

  /*< private >*/
  gpointer _reserved[16];
};

FOUNDRY_AVAILABLE_IN_ALL
char                *foundry_vcs_commit_dup_id        (FoundryVcsCommit *self);
FOUNDRY_AVAILABLE_IN_ALL
char                *foundry_vcs_commit_dup_title     (FoundryVcsCommit *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryVcsSignature *foundry_vcs_commit_dup_author    (FoundryVcsCommit *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryVcsSignature *foundry_vcs_commit_dup_committer (FoundryVcsCommit *self);
FOUNDRY_AVAILABLE_IN_ALL
guint                foundry_vcs_commit_get_n_parents (FoundryVcsCommit *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_vcs_commit_load_parent   (FoundryVcsCommit *self,
                                                       guint             index);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_vcs_commit_load_tree     (FoundryVcsCommit *self);

G_END_DECLS
