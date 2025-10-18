/* foundry-git-reference.c
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
#include "foundry-git-commit-private.h"
#include "foundry-git-error.h"
#include "foundry-git-reference-private.h"
#include "foundry-util.h"

struct _FoundryGitReference
{
  FoundryVcsReference  parent_instance;
  GMutex               mutex;
  git_reference       *reference;
};

G_DEFINE_FINAL_TYPE (FoundryGitReference, foundry_git_reference, FOUNDRY_TYPE_VCS_REFERENCE)

static char *
foundry_git_reference_dup_id (FoundryVcsReference *reference)
{
  FoundryGitReference *self = FOUNDRY_GIT_REFERENCE (reference);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);
  const git_oid *oid;

  if (git_reference_type (self->reference) == GIT_REFERENCE_SYMBOLIC)
    return g_strdup (git_reference_symbolic_target (self->reference));

  if ((oid = git_reference_target (self->reference)))
    {
      char str[GIT_OID_HEXSZ + 1];

      git_oid_tostr (str, sizeof str, oid);
      str[GIT_OID_HEXSZ] = 0;

      return g_strdup (str);
    }

  return NULL;
}

static char *
foundry_git_reference_dup_title (FoundryVcsReference *reference)
{
  FoundryGitReference *self = FOUNDRY_GIT_REFERENCE (reference);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);

  return g_strdup (git_reference_name (self->reference));
}

static gboolean
foundry_git_reference_is_symbolic (FoundryVcsReference *reference)
{
  FoundryGitReference *self = FOUNDRY_GIT_REFERENCE (reference);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);

  return git_reference_type (self->reference) == GIT_REFERENCE_SYMBOLIC;
}

static DexFuture *
foundry_git_reference_resolve_thread (gpointer data)
{
  FoundryGitReference *self = FOUNDRY_GIT_REFERENCE (data);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);

  if (git_reference_type (self->reference) == GIT_REFERENCE_SYMBOLIC)
    {
      g_autoptr(git_reference) resolved = NULL;

      if (git_reference_resolve (&resolved, self->reference) != 0)
        return foundry_git_reject_last_error ();

      return dex_future_new_take_object (_foundry_git_reference_new (g_steal_pointer (&resolved)));
    }

  return foundry_future_new_not_supported ();
}

static DexFuture *
foundry_git_reference_resolve (FoundryVcsReference *reference)
{
  g_assert (FOUNDRY_IS_GIT_REFERENCE (reference));

  return dex_thread_spawn ("[git-reference-resolve]",
                           foundry_git_reference_resolve_thread,
                           g_object_ref (reference),
                           g_object_unref);
}

static DexFuture *
foundry_git_reference_load_commit_thread (gpointer data)
{
  FoundryGitReference *self = FOUNDRY_GIT_REFERENCE (data);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);
  g_autoptr(git_object) object = NULL;

  if (git_reference_peel (&object, self->reference, GIT_OBJECT_COMMIT) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_take_object (_foundry_git_commit_new ((git_commit *)g_steal_pointer (&object),
                                                              (GDestroyNotify) git_object_free));
}

static DexFuture *
foundry_git_reference_load_commit (FoundryVcsReference *reference)
{
  g_assert (FOUNDRY_IS_GIT_REFERENCE (reference));

  return dex_thread_spawn ("[git-reference-commit]",
                           foundry_git_reference_load_commit_thread,
                           g_object_ref (reference),
                           g_object_unref);
}

static void
foundry_git_reference_finalize (GObject *object)
{
  FoundryGitReference *self = (FoundryGitReference *)object;

  g_clear_pointer (&self->reference, git_reference_free);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_git_reference_parent_class)->finalize (object);
}

static void
foundry_git_reference_class_init (FoundryGitReferenceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsReferenceClass *vcs_ref_class = FOUNDRY_VCS_REFERENCE_CLASS (klass);

  object_class->finalize = foundry_git_reference_finalize;

  vcs_ref_class->dup_id = foundry_git_reference_dup_id;
  vcs_ref_class->dup_title = foundry_git_reference_dup_title;
  vcs_ref_class->is_symbolic = foundry_git_reference_is_symbolic;
  vcs_ref_class->resolve = foundry_git_reference_resolve;
  vcs_ref_class->load_commit = foundry_git_reference_load_commit;
}

static void
foundry_git_reference_init (FoundryGitReference *self)
{
  g_mutex_init (&self->mutex);
}

/**
 * _foundry_git_reference_new:
 * @reference: (transfer full): the git_reference to wrap
 *
 * Creates a new [class@Foundry.GitReference] taking ownership of @reference.
 *
 * Returns: (transfer full):
 */
FoundryGitReference *
_foundry_git_reference_new (git_reference *reference)
{
  FoundryGitReference *self;

  g_return_val_if_fail (reference != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_REFERENCE, NULL);
  self->reference = g_steal_pointer (&reference);

  return self;
}
