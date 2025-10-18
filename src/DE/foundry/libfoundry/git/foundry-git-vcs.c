/* foundry-git-vcs.c
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

#include "foundry-auth-provider.h"
#include "foundry-git-autocleanups.h"
#include "foundry-git-file.h"
#include "foundry-git-file-list-private.h"
#include "foundry-git-error.h"
#include "foundry-git-blame-private.h"
#include "foundry-git-branch-private.h"
#include "foundry-git-file-private.h"
#include "foundry-git-reference-private.h"
#include "foundry-git-repository-private.h"
#include "foundry-git-remote-private.h"
#include "foundry-git-tag-private.h"
#include "foundry-git-tree.h"
#include "foundry-git-vcs-private.h"
#include "foundry-operation.h"
#include "foundry-util.h"

struct _FoundryGitVcs
{
  FoundryVcs            parent_instance;
  FoundryGitMonitor    *monitor;
  FoundryGitRepository *repository;
  GFile                *workdir;
};

G_DEFINE_FINAL_TYPE (FoundryGitVcs, foundry_git_vcs, FOUNDRY_TYPE_VCS)

static char *
foundry_git_vcs_dup_id (FoundryVcs *vcs)
{
  return g_strdup ("git");
}

static char *
foundry_git_vcs_dup_name (FoundryVcs *vcs)
{
  return g_strdup (_("Git"));
}

static char *
foundry_git_vcs_dup_branch_name (FoundryVcs *vcs)
{
  return _foundry_git_repository_dup_branch_name (FOUNDRY_GIT_VCS (vcs)->repository);
}

static guint
foundry_git_vcs_get_priority (FoundryVcs *vcs)
{
  return 100;
}

static gboolean
foundry_git_vcs_is_ignored (FoundryVcs *vcs,
                            const char *relative_path)
{
  FoundryGitVcs *self = FOUNDRY_GIT_VCS (vcs);

  return _foundry_git_repository_is_ignored (self->repository, relative_path);
}

static gboolean
foundry_git_vcs_is_file_ignored (FoundryVcs *vcs,
                                 GFile      *file)
{
  FoundryGitVcs *self = FOUNDRY_GIT_VCS (vcs);
  g_autofree char *relative_path = NULL;

  if (!g_file_has_prefix (file, self->workdir))
    return FALSE;

  relative_path = g_file_get_relative_path (self->workdir, file);

  return _foundry_git_repository_is_ignored (self->repository, relative_path);
}

static DexFuture *
foundry_git_vcs_list_files (FoundryVcs *vcs)
{
  return _foundry_git_repository_list_files (FOUNDRY_GIT_VCS (vcs)->repository);
}

static DexFuture *
foundry_git_vcs_list_branches (FoundryVcs *vcs)
{
  return _foundry_git_repository_list_branches (FOUNDRY_GIT_VCS (vcs)->repository);
}

static DexFuture *
foundry_git_vcs_list_tags (FoundryVcs *vcs)
{
  return _foundry_git_repository_list_tags (FOUNDRY_GIT_VCS (vcs)->repository);
}

static DexFuture *
foundry_git_vcs_list_remotes (FoundryVcs *vcs)
{
  return _foundry_git_repository_list_remotes (FOUNDRY_GIT_VCS (vcs)->repository);
}

static DexFuture *
foundry_git_vcs_find_file (FoundryVcs *vcs,
                           GFile      *file)
{
  return _foundry_git_repository_find_file (FOUNDRY_GIT_VCS (vcs)->repository, file);
}

static DexFuture *
foundry_git_vcs_find_remote (FoundryVcs *vcs,
                             const char *name)
{
  return _foundry_git_repository_find_remote (FOUNDRY_GIT_VCS (vcs)->repository, name);
}

static DexFuture *
foundry_git_vcs_find_commit (FoundryVcs *vcs,
                             const char *id)
{
  return _foundry_git_repository_find_commit (FOUNDRY_GIT_VCS (vcs)->repository, id);
}

static DexFuture *
foundry_git_vcs_find_tree (FoundryVcs *vcs,
                           const char *id)
{
  return _foundry_git_repository_find_tree (FOUNDRY_GIT_VCS (vcs)->repository, id);
}

static DexFuture *
foundry_git_vcs_blame (FoundryVcs     *vcs,
                       FoundryVcsFile *file,
                       GBytes         *bytes)
{
  FoundryGitVcs *self = FOUNDRY_GIT_VCS (vcs);
  g_autofree char *relative_path = foundry_vcs_file_dup_relative_path (file);

  return _foundry_git_repository_blame (self->repository, relative_path, bytes);
}

static DexFuture *
foundry_git_vcs_diff (FoundryVcs     *vcs,
                      FoundryVcsTree *tree_a,
                      FoundryVcsTree *tree_b)
{
  FoundryGitVcs *self = (FoundryGitVcs *)vcs;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_VCS (self));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_TREE (tree_a));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_TREE (tree_b));

  return _foundry_git_repository_diff (self->repository,
                                       FOUNDRY_GIT_TREE (tree_a),
                                       FOUNDRY_GIT_TREE (tree_b));
}

static DexFuture *
foundry_git_vcs_fetch (FoundryVcs       *vcs,
                       FoundryVcsRemote *remote,
                       FoundryOperation *operation)
{
  g_autoptr(FoundryAuthProvider) auth_provider = foundry_operation_dup_auth_provider (operation);

  if (auth_provider == NULL)
    {
      g_autoptr(FoundryContext) context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (vcs));
      auth_provider = foundry_auth_provider_new_for_context (context);
    }

  return _foundry_git_repository_fetch (FOUNDRY_GIT_VCS (vcs)->repository, auth_provider, remote, operation);
}

static DexFuture *
foundry_git_vcs_list_commits_with_file (FoundryVcs     *vcs,
                                        FoundryVcsFile *file)
{
  FoundryGitVcs *self = (FoundryGitVcs *)vcs;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_VCS (self));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_FILE (file));

  return _foundry_git_repository_list_commits_with_file (self->repository, file);
}

static DexFuture *
foundry_git_vcs_describe_line_changes (FoundryVcs     *vcs,
                                       FoundryVcsFile *file,
                                       GBytes         *contents)
{
  FoundryGitVcs *self = (FoundryGitVcs *)vcs;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_VCS (self));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_FILE (file));
  dex_return_error_if_fail (contents != NULL);

  return _foundry_git_repository_describe_line_changes (self->repository, file, contents);
}

static DexFuture *
foundry_git_vcs_query_file_status (FoundryVcs *vcs,
                                   GFile      *file)
{
  FoundryGitVcs *self = FOUNDRY_GIT_VCS (vcs);

  return _foundry_git_repository_query_file_status (self->repository, file);
}

static void
foundry_git_vcs_finalize (GObject *object)
{
  FoundryGitVcs *self = (FoundryGitVcs *)object;

  g_clear_object (&self->monitor);
  g_clear_object (&self->repository);
  g_clear_object (&self->workdir);

  G_OBJECT_CLASS (foundry_git_vcs_parent_class)->finalize (object);
}

static void
foundry_git_vcs_class_init (FoundryGitVcsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsClass *vcs_class = FOUNDRY_VCS_CLASS (klass);

  object_class->finalize = foundry_git_vcs_finalize;

  vcs_class->blame = foundry_git_vcs_blame;
  vcs_class->dup_branch_name = foundry_git_vcs_dup_branch_name;
  vcs_class->dup_id = foundry_git_vcs_dup_id;
  vcs_class->dup_name = foundry_git_vcs_dup_name;
  vcs_class->fetch = foundry_git_vcs_fetch;
  vcs_class->find_commit = foundry_git_vcs_find_commit;
  vcs_class->find_tree = foundry_git_vcs_find_tree;
  vcs_class->find_file = foundry_git_vcs_find_file;
  vcs_class->find_remote = foundry_git_vcs_find_remote;
  vcs_class->get_priority = foundry_git_vcs_get_priority;
  vcs_class->is_file_ignored = foundry_git_vcs_is_file_ignored;
  vcs_class->is_ignored = foundry_git_vcs_is_ignored;
  vcs_class->list_branches = foundry_git_vcs_list_branches;
  vcs_class->list_files = foundry_git_vcs_list_files;
  vcs_class->list_remotes = foundry_git_vcs_list_remotes;
  vcs_class->list_tags = foundry_git_vcs_list_tags;
  vcs_class->list_commits_with_file = foundry_git_vcs_list_commits_with_file;
  vcs_class->diff = foundry_git_vcs_diff;
  vcs_class->describe_line_changes = foundry_git_vcs_describe_line_changes;
  vcs_class->query_file_status = foundry_git_vcs_query_file_status;
}

static void
foundry_git_vcs_init (FoundryGitVcs *self)
{
}

static void
foundry_git_vcs_changed_cb (FoundryGitVcs     *self,
                            FoundryGitMonitor *monitor)
{
  g_assert (FOUNDRY_IS_GIT_VCS (self));
  g_assert (FOUNDRY_IS_GIT_MONITOR (monitor));

  g_object_notify (G_OBJECT (self), "branch-name");
}

static DexFuture *
foundry_git_vcs_setup_monitor_cb (DexFuture *future,
                                  gpointer   user_data)
{
  FoundryGitVcs *self = user_data;

  g_assert (FOUNDRY_IS_GIT_VCS (self));

  if ((self->monitor = dex_await_object (dex_ref (future), NULL)))
    g_signal_connect_object (self->monitor,
                             "changed",
                             G_CALLBACK (foundry_git_vcs_changed_cb),
                             self,
                             G_CONNECT_SWAPPED);

  return dex_future_new_true ();
}

DexFuture *
_foundry_git_vcs_new (FoundryContext *context,
                      git_repository *repository)
{
  FoundryGitVcs *self;
  DexFuture *future;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (repository != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_VCS,
                       "context", context,
                       NULL);

  self->workdir = g_file_new_for_path (git_repository_workdir (repository));
  self->repository = _foundry_git_repository_new (g_steal_pointer (&repository));

  future = _foundry_git_repository_create_monitor (self->repository);
  future = dex_future_then (future,
                            foundry_git_vcs_setup_monitor_cb,
                            g_object_ref (self),
                            g_object_unref);
  future = dex_future_then (future,
                            foundry_future_return_object,
                            g_object_ref (self),
                            g_object_unref);

  return g_steal_pointer (&future);
}

typedef struct _Initialize
{
  GFile *directory;
  guint bare : 1;
} Initialize;

static void
initialize_free (Initialize *state)
{
  g_clear_object (&state->directory);
  g_free (state);
}

static DexFuture *
foundry_git_initialize_thread (gpointer data)
{
  Initialize *state = data;
  g_autoptr(git_repository) repository = NULL;
  g_autofree char *path = NULL;

  g_assert (state != NULL);
  g_assert (G_IS_FILE (state->directory));
  g_assert (g_file_is_native (state->directory));

  path = g_file_get_path (state->directory);

  if (git_repository_init (&repository, path, state->bare) != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_true ();
}

/**
 * foundry_git_initialize:
 *
 * Initializes a new git repository.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value if successful or rejects with error.
 */
DexFuture *
foundry_git_initialize (GFile    *directory,
                        gboolean  bare)
{
  Initialize *state;

  dex_return_error_if_fail (G_IS_FILE (directory));
  dex_return_error_if_fail (g_file_is_native (directory));

  state = g_new0 (Initialize, 1);
  state->directory = g_object_ref (directory);
  state->bare = !!bare;

  return dex_thread_spawn ("[git-initialize]",
                           foundry_git_initialize_thread,
                           state,
                           (GDestroyNotify) initialize_free);

}

/**
 * foundry_git_vcs_list_status:
 * @self: a [class@Foundry.GitVcs]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.GitStatusEntry].
 */
DexFuture *
foundry_git_vcs_list_status (FoundryGitVcs *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_VCS (self));

  return _foundry_git_repository_list_status (self->repository);
}

/**
 * foundry_git_vcs_stage_entry:
 * @self: a [class@Foundry.GitVcs]
 * @entry: a [class@Foundry.GitStatusEntry]
 * @contents: (nullable): optional contents to use instead of what is in
 *   the working tree.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any value
 *   or rejects with error.
 */
DexFuture *
foundry_git_vcs_stage_entry (FoundryGitVcs         *self,
                             FoundryGitStatusEntry *entry,
                             GBytes                *contents)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_VCS (self));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_STATUS_ENTRY (entry));

  return _foundry_git_repository_stage_entry (self->repository, entry, contents);
}

/**
 * foundry_git_vcs_unstage_entry:
 * @self: a [class@Foundry.GitVcs]
 * @entry: a [class@Foundry.GitStatusEntry]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any value
 *   or rejects with error.
 */
DexFuture *
foundry_git_vcs_unstage_entry (FoundryGitVcs         *self,
                               FoundryGitStatusEntry *entry)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_VCS (self));
  dex_return_error_if_fail (FOUNDRY_IS_GIT_STATUS_ENTRY (entry));

  return _foundry_git_repository_unstage_entry (self->repository, entry);
}

/**
 * foundry_git_vcs_commit:
 * @self: a [class@Foundry.GitVcs]
 * @message:
 * @author_name: (nullable):
 * @author_email: (nullable):
 *
 * Simple API to create a new commit from the index.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Foundry.GitCommit] or rejects with error.
 */
DexFuture *
foundry_git_vcs_commit (FoundryGitVcs *self,
                        const char    *message,
                        const char    *author_name,
                        const char    *author_email)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_VCS (self));
  dex_return_error_if_fail (message != NULL);

  return _foundry_git_repository_commit (self->repository, message, author_name, author_email);
}
