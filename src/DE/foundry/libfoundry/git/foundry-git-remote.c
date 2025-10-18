/* foundry-git-remote.c
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

#include "foundry-git-remote-private.h"

struct _FoundryGitRemote
{
  FoundryVcsRemote  parent_instance;
  GMutex            mutex;
  git_remote       *remote;
  char             *name;
};

G_DEFINE_FINAL_TYPE (FoundryGitRemote, foundry_git_remote, FOUNDRY_TYPE_VCS_REMOTE)

static char *
foundry_git_remote_dup_name (FoundryVcsRemote *remote)
{
  return g_strdup (FOUNDRY_GIT_REMOTE (remote)->name);
}

static void
foundry_git_remote_finalize (GObject *object)
{
  FoundryGitRemote *self = (FoundryGitRemote *)object;

  g_clear_pointer (&self->remote, git_remote_free);
  g_clear_pointer (&self->name, g_free);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_git_remote_parent_class)->finalize (object);
}

static void
foundry_git_remote_class_init (FoundryGitRemoteClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsRemoteClass *vcs_remote_class = FOUNDRY_VCS_REMOTE_CLASS (klass);

  object_class->finalize = foundry_git_remote_finalize;

  vcs_remote_class->dup_name = foundry_git_remote_dup_name;
}

static void
foundry_git_remote_init (FoundryGitRemote *self)
{
  g_mutex_init (&self->mutex);
}

/**
 * _foundry_git_remote_new:
 * @remote: (transfer full): the git_remote to wrap
 * @name: (nullable): alternate name for the remote
 *
 * Creates a new [class@Foundry.GitRemote] taking ownership of @remote.
 *
 * Returns: (transfer full):
 */
FoundryGitRemote *
_foundry_git_remote_new (git_remote *remote,
                         const char *name)
{
  FoundryGitRemote *self;

  g_return_val_if_fail (remote != NULL, NULL);

  if (name == NULL)
    name = git_remote_name (remote);

  self = g_object_new (FOUNDRY_TYPE_GIT_REMOTE, NULL);
  self->remote = g_steal_pointer (&remote);
  self->name = g_strdup (name);

  return self;
}
