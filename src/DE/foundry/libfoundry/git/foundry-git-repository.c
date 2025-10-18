/* foundry-git-repository.c
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

#include <glib/gi18n-lib.h>
#include <glib/gstdio.h>

#include "foundry-auth-provider.h"
#include "foundry-git-autocleanups.h"
#include "foundry-git-blame-private.h"
#include "foundry-git-branch-private.h"
#include "foundry-git-callbacks-private.h"
#include "foundry-git-commit-private.h"
#include "foundry-git-error.h"
#include "foundry-git-file-list-private.h"
#include "foundry-git-file-private.h"
#include "foundry-git-line-changes-private.h"
#include "foundry-git-monitor-private.h"
#include "foundry-git-remote-private.h"
#include "foundry-git-repository-private.h"
#include "foundry-git-status-list-private.h"
#include "foundry-git-tag-private.h"
#include "foundry-git-tree-private.h"
#include "foundry-util.h"
#include "foundry-vcs.h"

#include "line-cache.h"

struct _FoundryGitRepository
{
  GObject         parent_instance;
  GMutex          mutex;
  git_repository *repository;
  GFile          *workdir;
  char           *git_dir;
  DexFuture      *monitor;
};

G_DEFINE_FINAL_TYPE (FoundryGitRepository, foundry_git_repository, G_TYPE_OBJECT)

static void
foundry_git_repository_finalize (GObject *object)
{
  FoundryGitRepository *self = (FoundryGitRepository *)object;

  dex_clear (&self->monitor);
  g_clear_pointer (&self->repository, git_repository_free);
  g_clear_pointer (&self->git_dir, g_free);
  g_clear_object (&self->workdir);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_git_repository_parent_class)->finalize (object);
}

static void
foundry_git_repository_class_init (FoundryGitRepositoryClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_git_repository_finalize;
}

static void
foundry_git_repository_init (FoundryGitRepository *self)
{
  g_mutex_init (&self->mutex);
}

/**
 * _foundry_git_repository_new:
 * @repository: (transfer full): the git_repository to wrap
 *
 * Creates a new [class@Foundry.GitRepository] taking ownership of @repository.
 *
 * Returns: (transfer full):
 */
FoundryGitRepository *
_foundry_git_repository_new (git_repository *repository)
{
  FoundryGitRepository *self;
  const char *path;

  g_return_val_if_fail (repository != NULL, NULL);

  path = git_repository_workdir (repository);

  self = g_object_new (FOUNDRY_TYPE_GIT_REPOSITORY, NULL);
  self->git_dir = g_strdup (git_repository_path (repository));
  self->repository = g_steal_pointer (&repository);
  self->workdir = g_file_new_for_path (path);

  return self;
}

static DexFuture *
foundry_git_repository_list_remotes_thread (gpointer data)
{
  FoundryGitRepository *self = data;
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(GListStore) store = NULL;
  g_auto(git_strarray) remotes = {0};

  g_assert (FOUNDRY_IS_GIT_REPOSITORY (self));

  locker = g_mutex_locker_new (&self->mutex);

  if (git_remote_list (&remotes, self->repository) != 0)
    return foundry_git_reject_last_error ();

  store = g_list_store_new (FOUNDRY_TYPE_VCS_REMOTE);

  for (gsize i = 0; i < remotes.count; i++)
    {
      g_autoptr(FoundryGitRemote) vcs_remote = NULL;
      g_autoptr(git_remote) remote = NULL;

      if (git_remote_lookup (&remote, self->repository, remotes.strings[i]) != 0)
        continue;

      if ((vcs_remote = _foundry_git_remote_new (g_steal_pointer (&remote), NULL)))
        g_list_store_append (store, vcs_remote);
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

DexFuture *
_foundry_git_repository_list_remotes (FoundryGitRepository *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));

  return dex_thread_spawn ("[git-list-remotes]",
                           foundry_git_repository_list_remotes_thread,
                           g_object_ref (self),
                           g_object_unref);
}

gboolean
_foundry_git_repository_is_ignored (FoundryGitRepository *self,
                                    const char           *relative_path)
{
  g_autoptr(GMutexLocker) locker = NULL;
  gboolean ignored = FALSE;

  g_return_val_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self), FALSE);
  g_return_val_if_fail (relative_path != NULL, FALSE);

  locker = g_mutex_locker_new (&self->mutex);

  if (git_ignore_path_is_ignored (&ignored, self->repository, relative_path) == GIT_OK)
    return ignored;

  return FALSE;
}

DexFuture *
_foundry_git_repository_list_files (FoundryGitRepository *self)
{
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(git_index) index = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));

  locker = g_mutex_locker_new (&self->mutex);

  if (git_repository_index (&index, self->repository) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_take_object (_foundry_git_file_list_new (self->workdir, g_steal_pointer (&index)));
}

typedef struct _Blame
{
  FoundryGitRepository *self;
  char                 *relative_path;
  GBytes               *bytes;
} Blame;

static void
blame_free (Blame *state)
{
  g_clear_object (&state->self);
  g_clear_pointer (&state->relative_path, g_free);
  g_clear_pointer (&state->bytes, g_bytes_unref);
  g_free (state);
}

static DexFuture *
foundry_git_repository_blame_thread (gpointer user_data)
{
  Blame *state = user_data;
  g_autoptr(git_blame) blame = NULL;
  g_autoptr(git_blame) bytes_blame = NULL;
  g_autoptr(GMutexLocker) locker = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_GIT_REPOSITORY (state->self));
  g_assert (state->relative_path != NULL);

  locker = g_mutex_locker_new (&state->self->mutex);

  if (git_blame_file (&blame, state->self->repository, state->relative_path, NULL) != 0)
    return foundry_git_reject_last_error ();

  if (state->bytes != NULL)
    {
      gconstpointer data = g_bytes_get_data (state->bytes, NULL);
      gsize size = g_bytes_get_size (state->bytes);

      if (git_blame_buffer (&bytes_blame, blame, data, size) != 0)
        return foundry_git_reject_last_error ();
    }

  return dex_future_new_take_object (_foundry_git_blame_new (g_steal_pointer (&blame),
                                                             g_steal_pointer (&bytes_blame)));
}

DexFuture *
_foundry_git_repository_blame (FoundryGitRepository *self,
                               const char           *relative_path,
                               GBytes               *bytes)
{
  Blame *state;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (relative_path != NULL);

  state = g_new0 (Blame, 1);
  state->self = g_object_ref (self);
  state->relative_path = g_strdup (relative_path);
  state->bytes = bytes ? g_bytes_ref (bytes) : NULL;

  return dex_thread_spawn ("[git-blame]",
                           foundry_git_repository_blame_thread,
                           state,
                           (GDestroyNotify) blame_free);
}

static DexFuture *
foundry_git_repository_list_branches_thread (gpointer data)
{
  FoundryGitRepository *self = data;
  g_autoptr(git_branch_iterator) iter = NULL;
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(GListStore) store = NULL;

  g_assert (FOUNDRY_IS_GIT_REPOSITORY (self));

  locker = g_mutex_locker_new (&self->mutex);

  if (git_branch_iterator_new (&iter, self->repository, GIT_BRANCH_ALL) < 0)
    return foundry_git_reject_last_error ();

  store = g_list_store_new (FOUNDRY_TYPE_VCS_BRANCH);

  for (;;)
    {
      g_autoptr(FoundryGitBranch) branch = NULL;
      g_autoptr(git_reference) ref = NULL;
      git_branch_t branch_type;

      if (git_branch_next (&ref, &branch_type, iter) != 0)
        break;

      if ((branch = _foundry_git_branch_new (self, g_steal_pointer (&ref), branch_type)))
        g_list_store_append (store, branch);
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

DexFuture *
_foundry_git_repository_list_branches (FoundryGitRepository *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));

  return dex_thread_spawn ("[git-list-branches]",
                           foundry_git_repository_list_branches_thread,
                           g_object_ref (self),
                           g_object_unref);
}

static DexFuture *
foundry_git_repository_list_tags_thread (gpointer data)
{
  FoundryGitRepository *self = data;
  g_autoptr(git_reference_iterator) iter = NULL;
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(GListStore) store = NULL;

  g_assert (FOUNDRY_IS_GIT_REPOSITORY (self));

  locker = g_mutex_locker_new (&self->mutex);

  if (git_reference_iterator_new (&iter, self->repository) < 0)
    return foundry_git_reject_last_error ();

  store = g_list_store_new (FOUNDRY_TYPE_VCS_TAG);

  for (;;)
    {
      g_autoptr(git_reference) ref = NULL;
      const char *name;

      if (git_reference_next (&ref, iter) != 0)
        break;

      if ((name = git_reference_name (ref)))
        {
          if (g_str_has_prefix (name, "refs/tags/") || !!strstr (name, "/tags/"))
            {
              g_autoptr(FoundryGitTag) tag = NULL;

              if ((tag = _foundry_git_tag_new (self, g_steal_pointer (&ref))))
                g_list_store_append (store, tag);
            }
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

DexFuture *
_foundry_git_repository_list_tags (FoundryGitRepository *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));

  return dex_thread_spawn ("[git-list-tags]",
                           foundry_git_repository_list_tags_thread,
                           g_object_ref (self),
                           g_object_unref);
}

typedef struct _FindRemote
{
  FoundryGitRepository *self;
  char *name;
} FindRemote;

static void
find_remote_free (FindRemote *state)
{
  g_clear_pointer (&state->name, g_free);
  g_clear_object (&state->self);
  g_free (state);
}

static DexFuture *
foundry_git_repository_find_remote_thread (gpointer data)
{
  FindRemote *state = data;
  g_autoptr(git_remote) remote = NULL;
  g_autoptr(GMutexLocker) locker = NULL;
  FoundryGitRepository *self;
  const char *name;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_GIT_REPOSITORY (state->self));
  g_assert (state->name != NULL);

  name = state->name;
  self = state->self;

  locker = g_mutex_locker_new (&self->mutex);

  if (git_remote_lookup (&remote, self->repository, name) == 0)
    {
      const char *alt_name = git_remote_name (remote);

      if (alt_name != NULL)
        name = alt_name;

      return dex_future_new_take_object (_foundry_git_remote_new (g_steal_pointer (&remote), name));
    }

  if (git_remote_create_anonymous (&remote, self->repository, name) == 0)
    {
      const char *alt_name = git_remote_name (remote);

      if (alt_name != NULL)
        name = alt_name;

      return dex_future_new_take_object (_foundry_git_remote_new (g_steal_pointer (&remote), name));
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

DexFuture *
_foundry_git_repository_find_remote (FoundryGitRepository *self,
                                     const char           *name)
{
  FindRemote *state;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (name != NULL);

  state = g_new0 (FindRemote, 1);
  state->self = g_object_ref (self);
  state->name = g_strdup (name);

  return dex_thread_spawn ("[git-find-remote]",
                           foundry_git_repository_find_remote_thread,
                           state,
                           (GDestroyNotify) find_remote_free);
}

DexFuture *
_foundry_git_repository_find_file (FoundryGitRepository *self,
                                   GFile                *file)
{
  g_autofree char *relative_path = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (G_IS_FILE (file));

  if (!g_file_has_prefix (file, self->workdir))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "File does not exist in working tree");

  relative_path = g_file_get_relative_path (self->workdir, file);

  g_assert (relative_path != NULL);

  return dex_future_new_take_object (_foundry_git_file_new (self->workdir, relative_path));
}

char *
_foundry_git_repository_dup_branch_name (FoundryGitRepository *self)
{
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(git_reference) head = NULL;

  g_return_val_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self), NULL);

  locker = g_mutex_locker_new (&self->mutex);

  if (git_repository_head (&head, self->repository) == 0)
    {
      const char *branch_name = NULL;

      if (git_branch_name (&branch_name, head) == 0)
        return g_strdup (branch_name);
    }

  return NULL;
}

typedef struct _Fetch
{
  char                *git_dir;
  char                *remote_name;
  FoundryOperation    *operation;
  FoundryAuthProvider *auth_provider;
  int                  pty_fd;
} Fetch;

static void
fetch_free (Fetch *state)
{
  g_clear_pointer (&state->git_dir, g_free);
  g_clear_pointer (&state->remote_name, g_free);
  g_clear_object (&state->operation);
  g_clear_object (&state->auth_provider);
  g_clear_fd (&state->pty_fd, NULL);
  g_free (state);
}

static DexFuture *
foundry_git_repository_fetch_thread (gpointer user_data)
{
  Fetch *state = user_data;
  g_autoptr(git_repository) repository = NULL;
  g_autoptr(git_remote) remote = NULL;
  git_fetch_options fetch_opts;
  int rval;

  g_assert (state != NULL);
  g_assert (state->git_dir != NULL);
  g_assert (state->remote_name != NULL);
  g_assert (FOUNDRY_IS_OPERATION (state->operation));

  if (git_repository_open (&repository, state->git_dir) != 0)
    return foundry_git_reject_last_error ();

  if (git_remote_lookup (&remote, repository, state->remote_name) != 0 &&
      git_remote_create_anonymous (&remote, repository, state->remote_name) != 0)
    return foundry_git_reject_last_error ();

  git_fetch_options_init (&fetch_opts, GIT_FETCH_OPTIONS_VERSION);

  fetch_opts.download_tags = GIT_REMOTE_DOWNLOAD_TAGS_ALL;
  fetch_opts.update_fetchhead = 1;

  _foundry_git_callbacks_init (&fetch_opts.callbacks, state->operation, state->auth_provider, state->pty_fd);
  rval = git_remote_fetch (remote, NULL, &fetch_opts, NULL);
  _foundry_git_callbacks_clear (&fetch_opts.callbacks);

  if (rval != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_true ();
}

DexFuture *
_foundry_git_repository_fetch (FoundryGitRepository *self,
                               FoundryAuthProvider  *auth_provider,
                               FoundryVcsRemote     *remote,
                               FoundryOperation     *operation)
{
  Fetch *state;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (FOUNDRY_IS_AUTH_PROVIDER (auth_provider));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_REMOTE (remote));
  dex_return_error_if_fail (FOUNDRY_IS_OPERATION (operation));

  state = g_new0 (Fetch, 1);
  state->remote_name = foundry_vcs_remote_dup_name (remote);
  state->git_dir = g_strdup (git_repository_path (self->repository));
  state->operation = g_object_ref (operation);
  state->auth_provider = g_object_ref (auth_provider);
  state->pty_fd = -1;

  return dex_thread_spawn ("[git-fetch]",
                           foundry_git_repository_fetch_thread,
                           state,
                           (GDestroyNotify) fetch_free);
}

typedef struct _FindByOid
{
  FoundryGitRepository *self;
  git_oid oid;
} FindByOid;

static void
find_by_oid_free (FindByOid *state)
{
  g_clear_object (&state->self);
  g_free (state);
}

static DexFuture *
foundry_git_repository_find_commit_thread (gpointer data)
{
  FindByOid *state = data;
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(git_commit) commit = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_GIT_REPOSITORY (state->self));

  locker = g_mutex_locker_new (&state->self->mutex);

  if (git_commit_lookup (&commit, state->self->repository, &state->oid) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_take_object (_foundry_git_commit_new (g_steal_pointer (&commit),
                                                              (GDestroyNotify) git_commit_free));
}

DexFuture *
_foundry_git_repository_find_commit (FoundryGitRepository *self,
                                     const char           *id)
{
  FindByOid *state;
  git_oid oid;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (id != NULL);

  if (git_oid_fromstr (&oid, id) != 0)
    return foundry_git_reject_last_error ();

  state = g_new0 (FindByOid, 1);
  state->self = g_object_ref (self);
  state->oid = oid;

  return dex_thread_spawn ("[git-find-commit]",
                           foundry_git_repository_find_commit_thread,
                           state,
                           (GDestroyNotify) find_by_oid_free);
}

static DexFuture *
foundry_git_repository_find_tree_thread (gpointer data)
{
  FindByOid *state = data;
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(git_tree) tree = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_GIT_REPOSITORY (state->self));

  locker = g_mutex_locker_new (&state->self->mutex);

  if (git_tree_lookup (&tree, state->self->repository, &state->oid) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_take_object (_foundry_git_tree_new (g_steal_pointer (&tree)));
}

DexFuture *
_foundry_git_repository_find_tree (FoundryGitRepository *self,
                                   const char           *id)
{
  FindByOid *state;
  git_oid oid;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (id != NULL);

  if (git_oid_fromstr (&oid, id) != 0)
    return foundry_git_reject_last_error ();

  state = g_new0 (FindByOid, 1);
  state->self = g_object_ref (self);
  state->oid = oid;

  return dex_thread_spawn ("[git-find-tree]",
                           foundry_git_repository_find_tree_thread,
                           state,
                           (GDestroyNotify) find_by_oid_free);
}

typedef struct _ListCommits
{
  char *git_dir;
  char *relative_path;
} ListCommits;

static void
list_commits_free (ListCommits *state)
{
  g_clear_pointer (&state->git_dir, g_free);
  g_clear_pointer (&state->relative_path, g_free);
  g_free (state);
}

static DexFuture *
foundry_git_repository_list_commits_thread (gpointer data)
{
  ListCommits *state = data;
  g_autoptr(git_repository) repository = NULL;
  g_autoptr(git_revwalk) walker = NULL;
  g_autoptr(GListStore) store = NULL;
  git_diff_options diff_opts = GIT_DIFF_OPTIONS_INIT;
  const char *paths[2] = {0};
  git_strarray pathspec = {(char**)paths, 1};
  git_oid oid;

  g_assert (state != NULL);
  g_assert (state->git_dir != NULL);
  g_assert (state->relative_path != NULL);

  store = g_list_store_new (FOUNDRY_TYPE_VCS_COMMIT);

  if (git_repository_open (&repository, state->git_dir) != 0)
    return foundry_git_reject_last_error ();

  if (git_revwalk_new (&walker, repository) != 0)
    return foundry_git_reject_last_error ();

  paths[0] = state->relative_path;
  diff_opts.pathspec = pathspec;

  git_revwalk_sorting (walker, GIT_SORT_TIME | GIT_SORT_REVERSE);
  git_revwalk_push_head (walker);

  /* This could be made faster if libgit2 had support for the Git bitmap index.
   * Without it, we cannot filter the tree by file. So instead, we have to walk
   * commits and compare them against the parent commit.
   *
   * This is mostly fine on smaller repositories, but can be more problematic
   * on larger ones.
   *
   * What would be nice is if someone went and added bitmap support to libgit2.
   */

  while (git_revwalk_next (&oid, walker) == 0)
    {
      g_autoptr(git_commit) commit = NULL;
      g_autoptr(git_commit) parent = NULL;
      g_autoptr(git_tree) parent_tree = NULL;
      g_autoptr(git_tree) commit_tree = NULL;
      g_autoptr(git_diff) diff = NULL;
      gsize n_deltas;

      if (git_commit_lookup (&commit, repository, &oid) != 0 ||
          git_commit_parentcount (commit) == 0 ||
          git_commit_parent (&parent, commit, 0) != 0 ||
          git_commit_tree (&commit_tree, commit) != 0 ||
          git_commit_tree (&parent_tree, parent) != 0)
        continue;

      if (git_diff_tree_to_tree (&diff, repository, parent_tree, commit_tree, &diff_opts) != 0)
        continue;

      n_deltas = git_diff_num_deltas (diff);

      for (gsize i = 0; i < n_deltas; i++)
        {
          const git_diff_delta *delta = git_diff_get_delta (diff, i);

          if (strcmp (delta->new_file.path, state->relative_path) == 0 ||
              strcmp (delta->old_file.path, state->relative_path) == 0)
            {
              g_autoptr(FoundryGitCommit) item = _foundry_git_commit_new (g_steal_pointer (&commit),
                                                                          (GDestroyNotify) git_commit_free);
              g_list_store_append (store, item);
              break;
            }
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

DexFuture *
_foundry_git_repository_list_commits_with_file (FoundryGitRepository *self,
                                                FoundryVcsFile       *file)
{
  ListCommits *state;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (FOUNDRY_IS_VCS_FILE (file));

  state = g_new0 (ListCommits, 1);
  state->relative_path = foundry_vcs_file_dup_relative_path (file);
  state->git_dir = g_strdup (self->git_dir);

  return dex_thread_spawn ("[git-list-commits]",
                           foundry_git_repository_list_commits_thread,
                           state,
                           (GDestroyNotify) list_commits_free);
}

DexFuture *
_foundry_git_repository_diff (FoundryGitRepository *self,
                              FoundryGitTree       *tree_a,
                              FoundryGitTree       *tree_b)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_TREE (tree_a));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_TREE (tree_b));

  return _foundry_git_tree_diff (tree_a, tree_b, self->git_dir);
}

typedef struct _DescribeLineChanges
{
  FoundryGitRepository *self;
  FoundryGitFile       *file;
  GBytes               *contents;
} DescribeLineChanges;

static void
describe_line_changes_free (DescribeLineChanges *state)
{
  g_clear_object (&state->self);
  g_clear_object (&state->file);
  g_clear_pointer (&state->contents, g_bytes_unref);
  g_free (state);
}

typedef struct _Range
{
  int old_start;
  int old_lines;
  int new_start;
  int new_lines;
} Range;

static int
diff_hunk_cb (const git_diff_delta *delta,
              const git_diff_hunk  *hunk,
              gpointer              user_data)
{
  GArray *ranges = user_data;
  Range range;

  g_assert (delta != NULL);
  g_assert (hunk != NULL);
  g_assert (ranges != NULL);

  range.old_start = hunk->old_start;
  range.old_lines = hunk->old_lines;
  range.new_start = hunk->new_start;
  range.new_lines = hunk->new_lines;

  g_array_append_val (ranges, range);

  return 0;
}

static DexFuture *
foundry_git_repository_describe_line_changes_fiber (gpointer data)
{
  DescribeLineChanges *state = data;
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(git_tree_entry) entry = NULL;
  g_autoptr(git_commit) commit = NULL;
  g_autoptr(git_blob) blob = NULL;
  g_autoptr(git_tree) tree = NULL;
  g_autoptr(GArray) ranges = NULL;
  g_autoptr(LineCache) cache = NULL;
  g_autofree char *path = NULL;
  FoundryGitRepository *self;
  FoundryGitFile *file;
  git_diff_options options;
  git_oid oid;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_GIT_REPOSITORY (state->self));
  g_assert (FOUNDRY_IS_GIT_FILE (state->file));
  g_assert (state->contents != NULL);

  self = state->self;
  file = state->file;
  path = foundry_vcs_file_dup_relative_path (FOUNDRY_VCS_FILE (file));
  ranges = g_array_new (FALSE, FALSE, sizeof (Range));

  locker = g_mutex_locker_new (&state->self->mutex);

  if (git_reference_name_to_id (&oid, self->repository, "HEAD") != 0)
    return foundry_git_reject_last_error ();

  if (git_commit_lookup (&commit, self->repository, &oid) != 0)
    return foundry_git_reject_last_error ();

  if (git_commit_tree (&tree, commit) != 0)
    return foundry_git_reject_last_error ();

  if (git_tree_entry_bypath (&entry, tree, path) != 0)
    return foundry_git_reject_last_error ();

  if (git_blob_lookup (&blob, self->repository, git_tree_entry_id (entry)) != 0)
    return foundry_git_reject_last_error ();

  git_diff_options_init (&options, GIT_DIFF_OPTIONS_VERSION);
  options.context_lines = 0;

  git_diff_blob_to_buffer (blob,
                           path,
                           g_bytes_get_data (state->contents, NULL),
                           g_bytes_get_size (state->contents),
                           path,
                           &options,
                           NULL,         /* File Callback */
                           NULL,         /* Binary Callback */
                           diff_hunk_cb, /* Hunk Callback */
                           NULL,
                           ranges);

  cache = line_cache_new ();

  for (guint i = 0; i < ranges->len; i++)
    {
      const Range *range = &g_array_index (ranges, Range, i);
      int start_line = range->new_start - 1;
      int end_line = range->new_start + range->new_lines - 1;

      if (range->old_lines == 0 && range->new_lines > 0)
        {
          line_cache_mark_range (cache, start_line, end_line, LINE_MARK_ADDED);
        }
      else if (range->new_lines == 0 && range->old_lines > 0)
        {
          if (start_line < 0)
            line_cache_mark_range (cache, 0, 0, LINE_MARK_PREVIOUS_REMOVED);
          else
            line_cache_mark_range (cache, start_line + 1, start_line + 1, LINE_MARK_REMOVED);
        }
      else
        {
          line_cache_mark_range (cache, start_line, end_line, LINE_MARK_CHANGED);
        }
    }

  return dex_future_new_take_object (_foundry_git_line_changes_new (g_steal_pointer (&cache)));
}

DexFuture *
_foundry_git_repository_describe_line_changes (FoundryGitRepository *self,
                                               FoundryVcsFile       *file,
                                               GBytes               *contents)
{
  DescribeLineChanges *state;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_FILE (file));
  dex_return_error_if_fail (contents != NULL);

  state = g_new0 (DescribeLineChanges, 1);
  state->self = g_object_ref (self);
  state->file = g_object_ref (FOUNDRY_GIT_FILE (file));
  state->contents = g_bytes_ref (contents);

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              foundry_git_repository_describe_line_changes_fiber,
                              state,
                              (GDestroyNotify) describe_line_changes_free);
}

typedef struct _QueryFileStatus
{
  FoundryGitRepository *self;
  DexPromise *promise;
  char *path;
} QueryFileStatus;

static void
foundry_git_repository_query_file_status_worker (gpointer data)
{
  QueryFileStatus *state = data;
  git_status_t status;
  int rval;

  g_mutex_lock (&state->self->mutex);
  rval = git_status_file (&status, state->self->repository, state->path);
  g_mutex_unlock (&state->self->mutex);

  if (rval != 0)
    {
      dex_promise_reject (state->promise,
                          g_error_new (G_IO_ERROR,
                                       G_IO_ERROR_INVAL,
                                       "Invalid parameter"));
    }
  else
    {
      FoundryVcsFileStatus flags = 0;
      g_auto(GValue) value = G_VALUE_INIT;

      if (status & GIT_STATUS_WT_NEW)
        flags |= FOUNDRY_VCS_FILE_STATUS_NEW_IN_TREE;

      if (status & GIT_STATUS_WT_MODIFIED)
        flags |= FOUNDRY_VCS_FILE_STATUS_MODIFIED_IN_TREE;

      if (status & GIT_STATUS_WT_DELETED)
        flags |= FOUNDRY_VCS_FILE_STATUS_DELETED_IN_TREE;

      if (status & GIT_STATUS_INDEX_NEW)
        flags |= FOUNDRY_VCS_FILE_STATUS_NEW_IN_STAGE;

      if (status & GIT_STATUS_INDEX_MODIFIED)
        flags |= FOUNDRY_VCS_FILE_STATUS_MODIFIED_IN_STAGE;

      if (status & GIT_STATUS_INDEX_DELETED)
        flags |= FOUNDRY_VCS_FILE_STATUS_DELETED_IN_STAGE;

      g_value_init (&value, FOUNDRY_TYPE_VCS_FILE_STATUS);
      g_value_set_flags (&value, flags);

      dex_promise_resolve (state->promise, &value);
    }

  g_clear_object (&state->self);
  g_clear_pointer (&state->path, g_free);
  dex_clear (&state->promise);
  g_free (state);
}

DexFuture *
_foundry_git_repository_query_file_status (FoundryGitRepository *self,
                                           GFile                *file)
{
  QueryFileStatus *state;
  DexPromise *promise;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (G_IS_FILE (file));

  if (!g_file_has_prefix (file, self->workdir))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Not found");

  promise = dex_promise_new ();

  state = g_new0 (QueryFileStatus, 1);
  state->self = g_object_ref (self);
  state->path = g_file_get_relative_path (self->workdir, file);
  state->promise = dex_ref (promise);

  dex_scheduler_push (dex_thread_pool_scheduler_get_default (),
                      foundry_git_repository_query_file_status_worker,
                      state);

  return DEX_FUTURE (promise);
}

static DexFuture *
foundry_git_repository_list_status_thread (gpointer data)
{
  const char *git_dir = data;
  g_autoptr(git_repository) repository = NULL;
  g_autoptr(git_status_list) status_list = NULL;
  git_status_options opts = GIT_STATUS_OPTIONS_INIT;

  g_assert (git_dir != NULL);

  if (git_repository_open (&repository, git_dir) != 0)
    return foundry_git_reject_last_error ();

  opts.show = GIT_STATUS_SHOW_INDEX_AND_WORKDIR;
  opts.flags = (GIT_STATUS_OPT_INCLUDE_UNTRACKED |
                GIT_STATUS_OPT_RENAMES_HEAD_TO_INDEX |
                GIT_STATUS_OPT_RECURSE_UNTRACKED_DIRS |
                GIT_STATUS_OPT_SORT_CASE_SENSITIVELY);

  if (git_status_list_new (&status_list, repository, &opts) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_take_object (_foundry_git_status_list_new (g_steal_pointer (&status_list)));
}

DexFuture *
_foundry_git_repository_list_status (FoundryGitRepository *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));

  return dex_thread_spawn ("[git-list-status]",
                           foundry_git_repository_list_status_thread,
                           g_strdup (self->git_dir),
                           g_free);
}

typedef struct _Stage
{
  FoundryGitRepository *self;
  FoundryGitStatusEntry *entry;
  GBytes *contents;
  char *git_dir;
} Stage;

static void
stage_free (Stage *state)
{
  g_clear_object (&state->self);
  g_clear_object (&state->entry);
  g_clear_pointer (&state->contents, g_bytes_unref);
  g_clear_pointer (&state->git_dir, g_free);
  g_free (state);
}

static DexFuture *
foundry_git_repository_stage_entry_thread (gpointer data)
{
  Stage *state = data;
  g_autoptr(git_repository) repository = NULL;
  g_autoptr(git_index) index = NULL;
  g_autofree char *path = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_GIT_REPOSITORY (state->self));
  g_assert (FOUNDRY_IS_GIT_STATUS_ENTRY (state->entry));

  path = foundry_git_status_entry_dup_path (state->entry);

  if (git_repository_open (&repository, state->git_dir) != 0)
    return foundry_git_reject_last_error ();

  if (git_repository_index (&index, repository) != 0)
    return foundry_git_reject_last_error ();

  if (state->contents == NULL)
    {
      if (git_index_add_bypath (index, path) != 0)
        return foundry_git_reject_last_error ();
    }
  else
    {
      git_index_entry entry;
      git_oid blob_oid;
      const char *buf;
      gsize buf_len;

      buf = g_bytes_get_data (state->contents, &buf_len);

      if (git_blob_create_from_buffer (&blob_oid, repository, buf, buf_len) != 0)
        return foundry_git_reject_last_error ();

      entry = (git_index_entry) {
        .mode = GIT_FILEMODE_BLOB,
        .id = blob_oid,
        .path = path,
      };

      if (git_index_add (index, &entry) != 0)
        return foundry_git_reject_last_error ();
    }

  if (git_index_write (index) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_true ();
}

DexFuture *
_foundry_git_repository_stage_entry (FoundryGitRepository  *self,
                                     FoundryGitStatusEntry *entry,
                                     GBytes                *contents)
{
  Stage *state;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));

  state = g_new0 (Stage, 1);
  state->self = g_object_ref (self);
  state->git_dir = g_strdup (self->git_dir);
  state->entry = g_object_ref (entry);
  state->contents = contents ? g_bytes_ref (contents) : NULL;

  return dex_thread_spawn ("[git-stage-entry]",
                           foundry_git_repository_stage_entry_thread,
                           state,
                           (GDestroyNotify) stage_free);
}

static DexFuture *
foundry_git_repository_unstage_entry_thread (gpointer data)
{
  FoundryPair *pair = data;
  FoundryGitRepository *self = FOUNDRY_GIT_REPOSITORY (pair->first);
  FoundryGitStatusEntry *entry = FOUNDRY_GIT_STATUS_ENTRY (pair->second);
  g_autofree char *path = foundry_git_status_entry_dup_path (entry);
  g_autoptr(git_repository) repository = NULL;
  g_autoptr(git_index) index = NULL;
  git_oid head_oid;
  int err;

  if (git_repository_open (&repository, self->git_dir) != 0)
    return foundry_git_reject_last_error ();

  if (git_repository_index (&index, repository) != 0)
    return foundry_git_reject_last_error ();

  if (git_reference_name_to_id (&head_oid, repository, "HEAD") != 0)
    {
      if (git_index_remove_bypath (index, path) != 0)
        return foundry_git_reject_last_error ();
    }
  else
    {
      g_autoptr(git_tree_entry) tree_entry = NULL;
      g_autoptr(git_reference) head_ref = NULL;
      g_autoptr(git_commit) head_commit = NULL;
      g_autoptr(git_tree) head_tree = NULL;

      if (git_repository_head (&head_ref, repository) != 0)
        return foundry_git_reject_last_error ();

      if (git_reference_peel ((git_object **)&head_commit, head_ref, GIT_OBJECT_COMMIT) != 0)
        return foundry_git_reject_last_error ();

      if (git_commit_tree (&head_tree, head_commit) != 0)
        return foundry_git_reject_last_error ();

      if ((err = git_tree_entry_bypath (&tree_entry, head_tree, path)))
        {
          if (err != GIT_ENOTFOUND)
            return foundry_git_reject_last_error ();

          if (git_index_remove_bypath (index, path) != 0)
            return foundry_git_reject_last_error ();
        }
      else
        {
          const git_index_entry ientry = {
            .path = path,
            .mode = git_tree_entry_filemode (tree_entry),
            .id = *git_tree_entry_id (tree_entry),
          };
          g_autoptr(git_blob) blob = NULL;
          const char *buf = NULL;
          gsize buf_len = 0;

          if (git_blob_lookup (&blob, repository, &ientry.id) == 0)
            {
              buf = git_blob_rawcontent (blob);
              buf_len = git_blob_rawsize (blob);
            }

          if (git_index_add_frombuffer (index, &ientry, buf, buf_len) != 0)
            return foundry_git_reject_last_error ();
        }
    }

  if (git_index_write (index) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_true ();
}

DexFuture *
_foundry_git_repository_unstage_entry (FoundryGitRepository  *self,
                                       FoundryGitStatusEntry *entry)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));

  return dex_thread_spawn ("[git-unstage-entry]",
                           foundry_git_repository_unstage_entry_thread,
                           foundry_pair_new (self, entry),
                           (GDestroyNotify) foundry_pair_free);
}

typedef struct _Commit
{
  char *git_dir;
  char *message;
  char *author_name;
  char *author_email;
} Commit;

static void
commit_free (Commit *state)
{
  g_clear_pointer (&state->git_dir, g_free);
  g_clear_pointer (&state->message, g_free);
  g_clear_pointer (&state->author_name, g_free);
  g_clear_pointer (&state->author_email, g_free);
  g_free (state);
}

static DexFuture *
foundry_git_repository_commit_thread (gpointer data)
{
  Commit *state = data;
  g_autofree char *author_name = NULL;
  g_autofree char *author_email = NULL;
  g_autoptr(git_repository) repository = NULL;
  g_autoptr(git_config) config = NULL;
  g_autoptr(git_index) index = NULL;
  g_autoptr(git_tree) tree = NULL;
  g_autoptr(git_signature) author = NULL;
  g_autoptr(git_signature) committer = NULL;
  g_autoptr(git_object) parent = NULL;
  g_autoptr(git_commit) commit = NULL;
  git_oid tree_oid;
  git_oid commit_oid;
  int err;

  g_assert (state != NULL);
  g_assert (state->git_dir != NULL);
  g_assert (state->message != NULL);

  if (git_repository_open (&repository, state->git_dir) != 0)
    return foundry_git_reject_last_error ();

  if (git_repository_config (&config, repository) != 0)
    return foundry_git_reject_last_error ();

  if (!g_set_str (&author_name, state->author_name))
    {
      g_autoptr(git_config_entry) entry = NULL;
      const char *real_name = g_get_real_name ();

      if (git_config_get_entry (&entry, config, "user.name") == 0)
        author_name = g_strdup (entry->value);
      else
        author_name = g_strdup (real_name ? real_name : g_get_user_name ());
    }

  if (!g_set_str (&author_email, state->author_email))
    {
      g_autoptr(git_config_entry) entry = NULL;

      if (git_config_get_entry (&entry, config, "user.email") == 0)
        author_email = g_strdup (entry->value);
      else
        author_email = g_strdup_printf ("%s@localhost", g_get_user_name ());
    }

  if (git_repository_index (&index, repository) != 0)
    return foundry_git_reject_last_error ();

  if (git_index_write_tree (&tree_oid, index) != 0)
    return foundry_git_reject_last_error ();

  if (git_tree_lookup (&tree, repository, &tree_oid) != 0)
    return foundry_git_reject_last_error ();

  if (git_signature_now (&author, author_name, author_email) != 0)
    return foundry_git_reject_last_error ();

  if (git_signature_dup (&committer, author) != 0)
    return foundry_git_reject_last_error ();

  if ((err = git_revparse_single (&parent, repository, "HEAD^{commit}")) != 0)
    {
      if (err != GIT_ENOTFOUND)
        return foundry_git_reject_last_error ();

      if (git_commit_create_v (&commit_oid, repository, "HEAD", author, committer, NULL, state->message, tree, 0) != 0)
        return foundry_git_reject_last_error ();
    }
  else
    {
      if (git_commit_create_v (&commit_oid, repository, "HEAD", author, committer, NULL, state->message, tree, 1, parent) != 0)
        return foundry_git_reject_last_error ();
    }

  if (git_commit_lookup (&commit, repository, &commit_oid) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_take_object (_foundry_git_commit_new (g_steal_pointer (&commit),
                                                              (GDestroyNotify) git_commit_free));
}

DexFuture *
_foundry_git_repository_commit (FoundryGitRepository *self,
                                const char           *message,
                                const char           *author_name,
                                const char           *author_email)
{
  Commit *state;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));
  dex_return_error_if_fail (message != NULL);

  state = g_new0 (Commit, 1);
  state->git_dir = g_strdup (self->git_dir);
  state->message = g_strdup (message);
  state->author_name = g_strdup (author_name);
  state->author_email = g_strdup (author_email);

  return dex_thread_spawn ("[git-commit]",
                           foundry_git_repository_commit_thread,
                           state,
                           (GDestroyNotify) commit_free);
}

/**
 * _foundry_git_repository_create_monitor:
 * @self: a [class@Foundry.GitRepository]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   #FoundryGitMonitor
 */
DexFuture *
_foundry_git_repository_create_monitor (FoundryGitRepository *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_REPOSITORY (self));

  if (self->monitor == NULL)
    self->monitor = foundry_git_monitor_new (self->git_dir);

  return dex_ref (self->monitor);
}
