/* foundry-git-commit.c
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
#include "foundry-git-signature-private.h"
#include "foundry-git-tree-private.h"

struct _FoundryGitCommit
{
  FoundryVcsCommit  parent_instance;
  GMutex            mutex;
  git_commit       *commit;
  GDestroyNotify    commit_destroy;
};

G_DEFINE_FINAL_TYPE (FoundryGitCommit, foundry_git_commit, FOUNDRY_TYPE_VCS_COMMIT)

static char *
foundry_git_commit_dup_id (FoundryVcsCommit *commit)
{
  FoundryGitCommit *self = FOUNDRY_GIT_COMMIT (commit);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);
  const git_oid *oid = git_commit_id (self->commit);
  char str[GIT_OID_HEXSZ + 1];

  if (oid == NULL)
    return NULL;

  git_oid_tostr (str, sizeof str, oid);
  str[GIT_OID_HEXSZ] = 0;

  return g_strdup (str);
}

static char *
foundry_git_commit_dup_title (FoundryVcsCommit *commit)
{
  FoundryGitCommit *self = FOUNDRY_GIT_COMMIT (commit);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);
  const char *message = git_commit_message (self->commit);
  const char *endline;

  if (message == NULL)
    return NULL;

  if ((endline = strchr (message, '\n')))
    return g_utf8_make_valid (message, endline - message);

  return g_utf8_make_valid (message, -1);
}

static FoundryVcsSignature *
foundry_git_commit_dup_author (FoundryVcsCommit *commit)
{
  FoundryGitCommit *self = FOUNDRY_GIT_COMMIT (commit);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);
  const git_signature *signature = git_commit_author (self->commit);
  g_autoptr(git_signature) copy = NULL;

  if (git_signature_dup (&copy, signature) == 0)
    return _foundry_git_signature_new (g_steal_pointer (&copy));

  return NULL;
}

static FoundryVcsSignature *
foundry_git_commit_dup_committer (FoundryVcsCommit *commit)
{
  FoundryGitCommit *self = FOUNDRY_GIT_COMMIT (commit);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);
  const git_signature *signature = git_commit_committer (self->commit);
  g_autoptr(git_signature) copy = NULL;

  if (git_signature_dup (&copy, signature) == 0)
    return _foundry_git_signature_new (g_steal_pointer (&copy));

  return NULL;
}

static guint
foundry_git_commit_get_n_parents (FoundryVcsCommit *commit)
{
  FoundryGitCommit *self = FOUNDRY_GIT_COMMIT (commit);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);

  return git_commit_parentcount (self->commit);
}

typedef struct _LoadParent
{
  FoundryGitCommit *self;
  guint index;
} LoadParent;

static void
load_parent_free (LoadParent *state)
{
  g_clear_object (&state->self);
  g_free (state);
}

static DexFuture *
foundry_git_commit_load_parent_thread (gpointer data)
{
  LoadParent *state = data;
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(git_commit) parent = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_GIT_COMMIT (state->self));

  locker = g_mutex_locker_new (&state->self->mutex);

  if (git_commit_parent (&parent, state->self->commit, state->index) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_take_object (_foundry_git_commit_new (g_steal_pointer (&parent),
                                                              (GDestroyNotify) git_commit_free));
}

static DexFuture *
foundry_git_commit_load_parent (FoundryVcsCommit *commit,
                                guint             index)
{
  FoundryGitCommit *self = (FoundryGitCommit *)commit;
  LoadParent *state;

  g_assert (FOUNDRY_IS_GIT_COMMIT (self));

  state = g_new0 (LoadParent, 1);
  state->self = g_object_ref (self);
  state->index = index;

  return dex_thread_spawn ("[git-load-parent]",
                           foundry_git_commit_load_parent_thread,
                           state,
                           (GDestroyNotify) load_parent_free);
}

static DexFuture *
foundry_git_commit_load_tree_thread (gpointer data)
{
  FoundryGitCommit *self = data;
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(git_tree) tree = NULL;

  g_assert (FOUNDRY_IS_GIT_COMMIT (self));

  locker = g_mutex_locker_new (&self->mutex);

  if (git_commit_tree (&tree, self->commit) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_take_object (_foundry_git_tree_new (g_steal_pointer (&tree)));
}

static DexFuture *
foundry_git_commit_load_tree (FoundryVcsCommit *commit)
{
  g_assert (FOUNDRY_IS_GIT_COMMIT (commit));

  return dex_thread_spawn ("[git-load-tree]",
                           foundry_git_commit_load_tree_thread,
                           g_object_ref (commit),
                           g_object_unref);
}

static void
foundry_git_commit_finalize (GObject *object)
{
  FoundryGitCommit *self = (FoundryGitCommit *)object;

  self->commit_destroy (self->commit);
  self->commit_destroy = NULL;
  self->commit = NULL;

  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_git_commit_parent_class)->finalize (object);
}

static void
foundry_git_commit_class_init (FoundryGitCommitClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsCommitClass *commit_class = FOUNDRY_VCS_COMMIT_CLASS (klass);

  object_class->finalize = foundry_git_commit_finalize;

  commit_class->dup_id =foundry_git_commit_dup_id;
  commit_class->dup_title =foundry_git_commit_dup_title;
  commit_class->dup_author = foundry_git_commit_dup_author;
  commit_class->dup_committer = foundry_git_commit_dup_committer;
  commit_class->get_n_parents = foundry_git_commit_get_n_parents;
  commit_class->load_parent = foundry_git_commit_load_parent;
  commit_class->load_tree = foundry_git_commit_load_tree;
}

static void
foundry_git_commit_init (FoundryGitCommit *self)
{
  g_mutex_init (&self->mutex);
}

/**
 * _foundry_git_commit_new:
 * @commit: (transfer full): the git_commit to wrap
 * @commit_destroy: destroy callback for @commit
 *
 * Creates a new [class@Foundry.GitCommit] taking ownership of @commit.
 *
 * Returns: (transfer full):
 */
FoundryGitCommit *
_foundry_git_commit_new (git_commit     *commit,
                         GDestroyNotify  commit_destroy)
{
  FoundryGitCommit *self;

  g_return_val_if_fail (commit != NULL, NULL);

  if (commit_destroy == NULL)
    commit_destroy = (GDestroyNotify)git_commit_free;

  self = g_object_new (FOUNDRY_TYPE_GIT_COMMIT, NULL);
  self->commit = g_steal_pointer (&commit);
  self->commit_destroy = commit_destroy;

  return self;
}
