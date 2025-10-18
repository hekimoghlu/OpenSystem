/* foundry-git-branch.c
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

#include "config.h"

#include "foundry-git-autocleanups.h"
#include "foundry-git-branch-private.h"
#include "foundry-git-error.h"
#include "foundry-git-reference-private.h"
#include "foundry-git-repository-private.h"

struct _FoundryGitBranch
{
  FoundryVcsBranch      parent_instance;
  GMutex                mutex;
  FoundryGitRepository *repository;
  git_reference        *reference;
  git_branch_t          branch_type;
};

G_DEFINE_FINAL_TYPE (FoundryGitBranch, foundry_git_branch, FOUNDRY_TYPE_VCS_BRANCH)

static char *
foundry_git_branch_dup_id (FoundryVcsBranch *branch)
{
  FoundryGitBranch *self = FOUNDRY_GIT_BRANCH (branch);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);
  const char *name = NULL;

  if (git_branch_name (&name, self->reference) == 0)
    return g_strdup (name);

  return NULL;
}

static char *
foundry_git_branch_dup_title (FoundryVcsBranch *branch)
{
  return foundry_git_branch_dup_id (branch);
}

static gboolean
foundry_git_branch_is_local (FoundryVcsBranch *branch)
{
  return FOUNDRY_GIT_BRANCH (branch)->branch_type == GIT_BRANCH_LOCAL;
}

static DexFuture *
foundry_git_branch_load_target (FoundryVcsBranch *branch)
{
  FoundryGitBranch *self = FOUNDRY_GIT_BRANCH (branch);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);

  if (git_reference_type (self->reference) == GIT_REFERENCE_SYMBOLIC)
    {
      g_autoptr(git_reference) resolved = NULL;

      if (git_reference_resolve (&resolved, self->reference) != 0)
        return foundry_git_reject_last_error ();

      return dex_future_new_take_object (_foundry_git_reference_new (g_steal_pointer (&resolved)));
    }
  else
    {
      g_autoptr(git_reference) copy = NULL;

      if (git_reference_dup (&copy, self->reference) != 0)
        return foundry_git_reject_last_error ();

      return dex_future_new_take_object (_foundry_git_reference_new (g_steal_pointer (&copy)));
    }
}

static void
foundry_git_branch_finalize (GObject *object)
{
  FoundryGitBranch *self = (FoundryGitBranch *)object;

  g_clear_pointer (&self->reference, git_reference_free);
  g_clear_object (&self->repository);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_git_branch_parent_class)->finalize (object);
}

static void
foundry_git_branch_class_init (FoundryGitBranchClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsBranchClass *vcs_branch_class = FOUNDRY_VCS_BRANCH_CLASS (klass);

  object_class->finalize = foundry_git_branch_finalize;

  vcs_branch_class->dup_id = foundry_git_branch_dup_id;
  vcs_branch_class->dup_title = foundry_git_branch_dup_title;
  vcs_branch_class->is_local = foundry_git_branch_is_local;
  vcs_branch_class->load_target = foundry_git_branch_load_target;
}

static void
foundry_git_branch_init (FoundryGitBranch *self)
{
  g_mutex_init (&self->mutex);
}

/**
 * _foundry_git_branch_new:
 * @reference: (transfer full): the git_reference to wrap
 *
 * Creates a new [class@Foundry.GitBranch] taking ownership of @reference.
 *
 * Returns: (transfer full):
 */
FoundryGitBranch *
_foundry_git_branch_new (FoundryGitRepository *repository,
                         git_reference        *reference,
                         git_branch_t          branch_type)
{
  FoundryGitBranch *self;

  g_return_val_if_fail (FOUNDRY_IS_GIT_REPOSITORY (repository), NULL);
  g_return_val_if_fail (reference != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_BRANCH, NULL);
  self->repository = g_object_ref (repository);
  self->reference = g_steal_pointer (&reference);
  self->branch_type = branch_type;

  return self;
}
