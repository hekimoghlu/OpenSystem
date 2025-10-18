/* foundry-git-tree.c
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
#include "foundry-git-diff-private.h"
#include "foundry-git-error.h"
#include "foundry-git-tree-private.h"

struct _FoundryGitTree
{
  FoundryVcsTree  parent_instance;
  GMutex          mutex;
  git_tree       *tree;
  git_oid         oid;
};

G_DEFINE_FINAL_TYPE (FoundryGitTree, foundry_git_tree, FOUNDRY_TYPE_VCS_TREE)

static char *
foundry_git_tree_dup_id (FoundryVcsTree *tree)
{
  char str[GIT_OID_HEXSZ + 1];
  git_oid_tostr (str, sizeof str, &FOUNDRY_GIT_TREE (tree)->oid);
  return g_strdup (str);
}

static void
foundry_git_tree_finalize (GObject *object)
{
  FoundryGitTree *self = (FoundryGitTree *)object;

  g_clear_pointer (&self->tree, git_tree_free);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_git_tree_parent_class)->finalize (object);
}

static void
foundry_git_tree_class_init (FoundryGitTreeClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsTreeClass *tree_class = FOUNDRY_VCS_TREE_CLASS (klass);

  object_class->finalize = foundry_git_tree_finalize;

  tree_class->dup_id = foundry_git_tree_dup_id;
}

static void
foundry_git_tree_init (FoundryGitTree *self)
{
  g_mutex_init (&self->mutex);
}

FoundryGitTree *
_foundry_git_tree_new (git_tree *tree)
{
  FoundryGitTree *self;

  g_return_val_if_fail (tree != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_TREE, NULL);
  self->oid = *git_tree_id (tree);
  self->tree = g_steal_pointer (&tree);

  return self;
}

typedef struct _Diff
{
  char *git_dir;
  git_oid tree_a;
  git_oid tree_b;
} Diff;

static void
diff_free (Diff *state)
{
  g_clear_pointer (&state->git_dir, g_free);
  g_free (state);
}

static DexFuture *
foundry_git_tree_diff_thread (gpointer data)
{
  Diff *state = data;
  g_autoptr(git_repository) repository = NULL;
  g_autoptr(git_tree) tree_a = NULL;
  g_autoptr(git_tree) tree_b = NULL;
  g_autoptr(git_diff) diff = NULL;

  g_assert (state != NULL);
  g_assert (state->git_dir != NULL);

  if (git_repository_open (&repository, state->git_dir) != 0 ||
      git_tree_lookup (&tree_a, repository, &state->tree_a) != 0 ||
      git_tree_lookup (&tree_b, repository, &state->tree_b) != 0 ||
      git_diff_tree_to_tree (&diff, repository, tree_a, tree_b, NULL) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_take_object (_foundry_git_diff_new (g_steal_pointer (&diff)));
}

DexFuture *
_foundry_git_tree_diff (FoundryGitTree *self,
                        FoundryGitTree *other,
                        const char     *git_dir)
{
  Diff *state;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_TREE (self));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_TREE (other));
  dex_return_error_if_fail (git_dir != NULL);

  /* TODO: we could trylock both self/other and avoid having to re-open
   * the repository to avoid potential deadlocks.
   */

  state = g_new0 (Diff, 1);
  state->git_dir = g_strdup (git_dir);
  state->tree_a = self->oid;
  state->tree_b = other->oid;

  return dex_thread_spawn ("[git-tree-diff]",
                           foundry_git_tree_diff_thread,
                           state,
                           (GDestroyNotify) diff_free);
}
