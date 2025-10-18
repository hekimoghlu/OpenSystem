/* foundry-git-blame.c
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
#include "foundry-git-blame-private.h"
#include "foundry-git-signature-private.h"
#include "foundry-vcs-file.h"

struct _FoundryGitBlame
{
  FoundryVcsBlame  parent_instance;
  GMutex           mutex;
  git_blame       *base_blame;
  git_blame       *bytes_blame;
};

G_DEFINE_FINAL_TYPE (FoundryGitBlame, foundry_git_blame, FOUNDRY_TYPE_VCS_BLAME)

static git_blame *
get_blame_locked (FoundryGitBlame *self)
{
  return self->bytes_blame ? self->bytes_blame : self->base_blame;
}

static DexFuture *
foundry_git_blame_update (FoundryVcsBlame *vcs_blame,
                          GBytes          *contents)
{
  FoundryGitBlame *self = (FoundryGitBlame *)vcs_blame;
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(git_blame) blame = NULL;
  gconstpointer data = NULL;
  gsize size = 0;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_BLAME (self));
  dex_return_error_if_fail (contents != NULL);

  locker = g_mutex_locker_new (&self->mutex);

  if (contents != NULL)
    data = g_bytes_get_data (contents, &size);

  g_clear_pointer (&self->bytes_blame, git_blame_free);

  if (git_blame_buffer (&blame, self->base_blame, data, size) == 0)
    self->bytes_blame = g_steal_pointer (&blame);

  return dex_future_new_true ();
}

static FoundryVcsSignature *
foundry_git_blame_query_line (FoundryVcsBlame *blame,
                              guint            line)
{
  FoundryGitBlame *self = (FoundryGitBlame *)blame;
  g_autoptr(GMutexLocker) locker = NULL;
  const git_blame_hunk *hunk;
  git_blame *gblame;

  g_assert (FOUNDRY_IS_GIT_BLAME (self));
  g_assert (self->base_blame != NULL);

  locker = g_mutex_locker_new (&self->mutex);
  gblame = get_blame_locked (self);

  if ((hunk = git_blame_get_hunk_byline (gblame, line + 1)))
    {
      g_autoptr(git_signature) copy = NULL;

      /* TODO: This is often accessed sequentially so it might make
       *       sense to keep the most-recently-used one and then
       *       reuse it instead of copy/construct on each line.
       */

      if (git_signature_dup (&copy, hunk->final_signature) != 0)
        return NULL;

      return _foundry_git_signature_new (g_steal_pointer (&copy));
    }

  return NULL;
}

static guint
foundry_git_blame_get_n_lines (FoundryVcsBlame *blame)
{
  FoundryGitBlame *self = (FoundryGitBlame *)blame;
  g_autoptr(GMutexLocker) locker = NULL;
  git_blame *gblame;
  gsize hunk_count;
  guint n_lines = 0;

  g_assert (FOUNDRY_IS_GIT_BLAME (self));
  g_assert (self->base_blame != NULL);

  locker = g_mutex_locker_new (&self->mutex);
  gblame = get_blame_locked (self);

  hunk_count = git_blame_get_hunk_count (gblame);

  for (gsize i = 0; i < hunk_count; i++)
    {
      const git_blame_hunk *hunk = git_blame_get_hunk_byindex (gblame, i);

      if (hunk != NULL)
        n_lines += hunk->lines_in_hunk;
    }

  return n_lines;
}

static void
foundry_git_blame_finalize (GObject *object)
{
  FoundryGitBlame *self = (FoundryGitBlame *)object;

  g_clear_pointer (&self->bytes_blame, git_blame_free);
  g_clear_pointer (&self->base_blame, git_blame_free);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_git_blame_parent_class)->finalize (object);
}

static void
foundry_git_blame_class_init (FoundryGitBlameClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsBlameClass *vcs_blame_class = FOUNDRY_VCS_BLAME_CLASS (klass);

  object_class->finalize = foundry_git_blame_finalize;

  vcs_blame_class->update = foundry_git_blame_update;
  vcs_blame_class->query_line = foundry_git_blame_query_line;
  vcs_blame_class->get_n_lines = foundry_git_blame_get_n_lines;
}

static void
foundry_git_blame_init (FoundryGitBlame *self)
{
  g_mutex_init (&self->mutex);
}

FoundryGitBlame *
_foundry_git_blame_new (git_blame *base_blame,
                        git_blame *bytes_blame)
{
  FoundryGitBlame *self;

  g_return_val_if_fail (base_blame != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_BLAME, NULL);
  self->base_blame = g_steal_pointer (&base_blame);
  self->bytes_blame = g_steal_pointer (&bytes_blame);

  return self;
}

