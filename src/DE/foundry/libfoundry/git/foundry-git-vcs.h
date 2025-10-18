/* foundry-git-vcs.h
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

#include "foundry-vcs.h"
#include "foundry-git-status-entry.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_GIT_VCS (foundry_git_vcs_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryGitVcs, foundry_git_vcs, FOUNDRY, GIT_VCS, FoundryVcs)

FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_git_initialize        (GFile                 *directory,
                                          gboolean               bare);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_git_vcs_list_status   (FoundryGitVcs         *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_git_vcs_stage_entry   (FoundryGitVcs         *self,
                                          FoundryGitStatusEntry *entry,
                                          GBytes                *contents);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_git_vcs_unstage_entry (FoundryGitVcs         *self,
                                          FoundryGitStatusEntry *entry);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_git_vcs_commit        (FoundryGitVcs         *self,
                                          const char            *message,
                                          const char            *author_name,
                                          const char            *author_email);

G_END_DECLS
