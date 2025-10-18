/* foundry-git-tag.c
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
#include "foundry-git-error.h"
#include "foundry-git-reference-private.h"
#include "foundry-git-tag-private.h"

struct _FoundryGitTag
{
  FoundryVcsTag         parent_instance;
  GMutex                mutex;
  FoundryGitRepository *repository;
  git_reference        *reference;
  guint                 is_local : 1;
};

G_DEFINE_FINAL_TYPE (FoundryGitTag, foundry_git_tag, FOUNDRY_TYPE_VCS_TAG)

static char *
foundry_git_tag_dup_id (FoundryVcsTag *tag)
{
  FoundryGitTag *self = FOUNDRY_GIT_TAG (tag);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);

  return g_strdup (git_reference_name (self->reference));
}

static char *
foundry_git_tag_dup_title (FoundryVcsTag *tag)
{
  FoundryGitTag *self = FOUNDRY_GIT_TAG (tag);
  g_autoptr(GMutexLocker) locker = g_mutex_locker_new (&self->mutex);
  const char *name = git_reference_name (self->reference);

  if (name == NULL)
    return NULL;

  if (g_str_has_prefix (name, "refs/tags/"))
    return g_strdup (name + strlen ("refs/tags/"));

  return g_strdup (name);
}

static gboolean
foundry_git_tag_is_local (FoundryVcsTag *tag)
{
  return FOUNDRY_GIT_TAG (tag)->is_local;
}

static DexFuture *
foundry_git_tag_load_target (FoundryVcsTag *tag)
{
  FoundryGitTag *self = FOUNDRY_GIT_TAG (tag);
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
foundry_git_tag_finalize (GObject *object)
{
  FoundryGitTag *self = (FoundryGitTag *)object;

  g_clear_pointer (&self->reference, git_reference_free);
  g_clear_object (&self->repository);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_git_tag_parent_class)->finalize (object);
}

static void
foundry_git_tag_class_init (FoundryGitTagClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsTagClass *vcs_tag_class = FOUNDRY_VCS_TAG_CLASS (klass);

  object_class->finalize = foundry_git_tag_finalize;

  vcs_tag_class->dup_id = foundry_git_tag_dup_id;
  vcs_tag_class->dup_title = foundry_git_tag_dup_title;
  vcs_tag_class->is_local = foundry_git_tag_is_local;
  vcs_tag_class->load_target = foundry_git_tag_load_target;
}

static void
foundry_git_tag_init (FoundryGitTag *self)
{
  g_mutex_init (&self->mutex);
}

/**
 * _foundry_git_tag_new:
 * @self: a [class@Foundry.GitRepository]
 * @reference: (transfer full): the git_reference to wrap
 *
 * Creates a new [class@Foundry.GitTag] taking ownership of @reference.
 *
 * Returns: (transfer full):
 */
FoundryGitTag *
_foundry_git_tag_new (FoundryGitRepository *repository,
                      git_reference        *reference)
{
  FoundryGitTag *self;
  const char *name;

  g_return_val_if_fail (FOUNDRY_IS_GIT_REPOSITORY (repository), NULL);
  g_return_val_if_fail (reference != NULL, NULL);

  name = git_reference_name (reference);

  self = g_object_new (FOUNDRY_TYPE_GIT_TAG, NULL);
  self->repository = g_object_ref (repository);
  self->reference = g_steal_pointer (&reference);
  self->is_local = name && g_str_has_prefix (name, "refs/tags/");

  return self;
}
