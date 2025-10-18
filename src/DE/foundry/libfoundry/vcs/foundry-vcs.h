/* foundry-vcs.h
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

#include "foundry-contextual.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_VCS             (foundry_vcs_get_type())
#define FOUNDRY_TYPE_VCS_FILE_STATUS (foundry_vcs_file_status_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryVcs, foundry_vcs, FOUNDRY, VCS, FoundryContextual)

struct _FoundryVcsClass
{
  FoundryContextualClass parent_class;

  char      *(*dup_id)                 (FoundryVcs       *self);
  char      *(*dup_name)               (FoundryVcs       *self);
  char      *(*dup_branch_name)        (FoundryVcs       *self);
  guint      (*get_priority)           (FoundryVcs       *self);
  gboolean   (*is_ignored)             (FoundryVcs       *self,
                                        const char       *relative_path);
  gboolean   (*is_file_ignored)        (FoundryVcs       *self,
                                        GFile            *file);
  DexFuture *(*list_files)             (FoundryVcs       *self);
  DexFuture *(*find_file)              (FoundryVcs       *self,
                                        GFile            *file);
  DexFuture *(*blame)                  (FoundryVcs       *self,
                                        FoundryVcsFile   *file,
                                        GBytes           *bytes);
  DexFuture *(*list_branches)          (FoundryVcs       *self);
  DexFuture *(*list_tags)              (FoundryVcs       *self);
  DexFuture *(*list_remotes)           (FoundryVcs       *self);
  DexFuture *(*find_remote)            (FoundryVcs       *self,
                                        const char       *name);
  DexFuture *(*find_commit)            (FoundryVcs       *self,
                                        const char       *id);
  DexFuture *(*find_tree)              (FoundryVcs       *self,
                                        const char       *id);
  DexFuture *(*fetch)                  (FoundryVcs       *self,
                                        FoundryVcsRemote *remote,
                                        FoundryOperation *operation);
  DexFuture *(*list_commits_with_file) (FoundryVcs       *self,
                                        FoundryVcsFile   *file);
  DexFuture *(*diff)                   (FoundryVcs       *self,
                                        FoundryVcsTree   *tree_a,
                                        FoundryVcsTree   *tree_b);
  DexFuture *(*describe_line_changes)  (FoundryVcs       *self,
                                        FoundryVcsFile   *file,
                                        GBytes           *contents);
  DexFuture *(*query_file_status)      (FoundryVcs       *vcs,
                                        GFile            *file);

  /*< private >*/
  gpointer _reserved[20];
};

FOUNDRY_AVAILABLE_IN_ALL
GType      foundry_vcs_file_status_get_type   (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_vcs_get_active             (FoundryVcs       *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_dup_id                 (FoundryVcs       *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_dup_name               (FoundryVcs       *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_dup_branch_name        (FoundryVcs       *self);
FOUNDRY_AVAILABLE_IN_ALL
guint      foundry_vcs_get_priority           (FoundryVcs       *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_vcs_is_ignored             (FoundryVcs       *self,
                                               const char       *relative_path);
FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_vcs_is_file_ignored        (FoundryVcs       *self,
                                               GFile            *file);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_list_files             (FoundryVcs       *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_find_file              (FoundryVcs       *self,
                                               GFile            *file);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_blame                  (FoundryVcs       *self,
                                               FoundryVcsFile   *file,
                                               GBytes           *bytes);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_list_branches          (FoundryVcs       *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_list_tags              (FoundryVcs       *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_list_remotes           (FoundryVcs       *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_fetch                  (FoundryVcs       *self,
                                               FoundryVcsRemote *remote,
                                               FoundryOperation *operation);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_find_remote            (FoundryVcs       *self,
                                               const char       *name);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_find_commit            (FoundryVcs       *self,
                                               const char       *id);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_find_tree              (FoundryVcs       *self,
                                               const char       *id);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_list_commits_with_file (FoundryVcs       *self,
                                               FoundryVcsFile   *file);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_diff                   (FoundryVcs       *self,
                                               FoundryVcsTree   *tree_a,
                                               FoundryVcsTree   *tree_b);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_describe_line_changes  (FoundryVcs       *self,
                                               FoundryVcsFile   *file,
                                               GBytes           *contents);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_query_file_status      (FoundryVcs       *self,
                                               GFile            *file);

G_END_DECLS
