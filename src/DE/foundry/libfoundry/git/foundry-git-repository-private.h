/* foundry-git-repository-private.h
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

#include <git2.h>
#include <glib-object.h>
#include <libdex.h>

#include "foundry-context.h"
#include "foundry-git-monitor-private.h"
#include "foundry-git-tree.h"
#include "foundry-operation.h"
#include "foundry-vcs-remote.h"
#include "foundry-git-status-entry.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_GIT_REPOSITORY (foundry_git_repository_get_type())

G_DECLARE_FINAL_TYPE (FoundryGitRepository, foundry_git_repository, FOUNDRY, GIT_REPOSITORY, GObject)

FoundryGitRepository *_foundry_git_repository_new                    (git_repository        *repository);
DexFuture            *_foundry_git_repository_create_monitor         (FoundryGitRepository  *self);
char                 *_foundry_git_repository_dup_branch_name        (FoundryGitRepository  *self);
DexFuture            *_foundry_git_repository_list_branches          (FoundryGitRepository  *self);
DexFuture            *_foundry_git_repository_list_tags              (FoundryGitRepository  *self);
DexFuture            *_foundry_git_repository_list_remotes           (FoundryGitRepository  *self);
DexFuture            *_foundry_git_repository_list_files             (FoundryGitRepository  *self);
DexFuture            *_foundry_git_repository_find_file              (FoundryGitRepository  *self,
                                                                      GFile                 *file);
gboolean              _foundry_git_repository_is_ignored             (FoundryGitRepository  *self,
                                                                      const char            *relative_path);
DexFuture            *_foundry_git_repository_blame                  (FoundryGitRepository  *self,
                                                                      const char            *relative_path,
                                                                      GBytes                *bytes);
DexFuture            *_foundry_git_repository_find_remote            (FoundryGitRepository  *self,
                                                                      const char            *name);
DexFuture            *_foundry_git_repository_fetch                  (FoundryGitRepository  *self,
                                                                      FoundryAuthProvider   *auth_provider,
                                                                      FoundryVcsRemote      *remote,
                                                                      FoundryOperation      *operation);
DexFuture            *_foundry_git_repository_find_commit            (FoundryGitRepository  *self,
                                                                      const char            *id);
DexFuture            *_foundry_git_repository_find_tree              (FoundryGitRepository  *self,
                                                                      const char            *id);
DexFuture            *_foundry_git_repository_list_commits_with_file (FoundryGitRepository  *self,
                                                                      FoundryVcsFile        *file);
DexFuture            *_foundry_git_repository_diff                   (FoundryGitRepository  *self,
                                                                      FoundryGitTree        *tree_a,
                                                                      FoundryGitTree        *tree_b);
DexFuture            *_foundry_git_repository_describe_line_changes  (FoundryGitRepository  *self,
                                                                      FoundryVcsFile        *file,
                                                                      GBytes                *contents);
DexFuture            *_foundry_git_repository_query_file_status      (FoundryGitRepository  *self,
                                                                      GFile                 *file);
DexFuture            *_foundry_git_repository_list_status            (FoundryGitRepository  *self);
DexFuture            *_foundry_git_repository_stage_entry            (FoundryGitRepository  *self,
                                                                      FoundryGitStatusEntry *entry,
                                                                      GBytes                *contents);
DexFuture            *_foundry_git_repository_unstage_entry          (FoundryGitRepository  *self,
                                                                      FoundryGitStatusEntry *entry);
DexFuture            *_foundry_git_repository_commit                 (FoundryGitRepository  *self,
                                                                      const char            *message,
                                                                      const char            *author_name,
                                                                      const char            *author_email);

G_END_DECLS
