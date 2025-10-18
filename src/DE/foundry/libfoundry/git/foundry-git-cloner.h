/* foundry-git-cloner.h
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

#define FOUNDRY_TYPE_GIT_CLONER (foundry_git_cloner_get_type())
#define FOUNDRY_GIT_CLONE_ERROR (foundry_git_clone_error_quark())

typedef enum _FoundryGitCloneError
{
  FOUNDRY_GIT_CLONE_ERROR_INVALID_URI,
  FOUNDRY_GIT_CLONE_ERROR_INVALID_DIRECTORY,
  FOUNDRY_GIT_CLONE_ERROR_INVALID_REMOTE_BRANCH_NAME,
} FoundryGitCloneError;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryGitCloner, foundry_git_cloner, FOUNDRY, GIT_CLONER, GObject)

FOUNDRY_AVAILABLE_IN_ALL
GQuark            foundry_git_clone_error_quark             (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryGitCloner *foundry_git_cloner_new                    (void);
FOUNDRY_AVAILABLE_IN_ALL
char             *foundry_git_cloner_dup_uri                (FoundryGitCloner  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_git_cloner_set_uri                (FoundryGitCloner  *self,
                                                             const char        *uri);
FOUNDRY_AVAILABLE_IN_ALL
gboolean          foundry_git_cloner_get_bare               (FoundryGitCloner  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_git_cloner_set_bare               (FoundryGitCloner  *self,
                                                             gboolean           bare);
FOUNDRY_AVAILABLE_IN_ALL
GFile            *foundry_git_cloner_dup_directory          (FoundryGitCloner  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_git_cloner_set_directory          (FoundryGitCloner  *self,
                                                             GFile             *directory);
FOUNDRY_AVAILABLE_IN_ALL
char             *foundry_git_cloner_dup_remote_branch_name (FoundryGitCloner  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_git_cloner_set_remote_branch_name (FoundryGitCloner  *self,
                                                             const char        *remote_branch_name);
FOUNDRY_AVAILABLE_IN_ALL
char             *foundry_git_cloner_dup_author_name        (FoundryGitCloner  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_git_cloner_set_author_name        (FoundryGitCloner  *self,
                                                             const char        *author_name);
FOUNDRY_AVAILABLE_IN_ALL
char             *foundry_git_cloner_dup_author_email       (FoundryGitCloner  *self);
FOUNDRY_AVAILABLE_IN_ALL
void              foundry_git_cloner_set_author_email       (FoundryGitCloner  *self,
                                                             const char        *author_email);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture        *foundry_git_cloner_validate               (FoundryGitCloner  *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture        *foundry_git_cloner_clone                  (FoundryGitCloner  *self,
                                                             int                pty_fd,
                                                             FoundryOperation  *operation);

G_END_DECLS
