/* foundry-git-delta.c
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

#include "foundry-git-delta-private.h"

struct _FoundryGitDelta
{
  GObject parent_instance;

  char *old_path;
  char *new_path;

  git_oid old_oid;
  git_oid new_oid;
};

G_DEFINE_FINAL_TYPE (FoundryGitDelta, foundry_git_delta, FOUNDRY_TYPE_VCS_DELTA)

static char *
foundry_git_delta_dup_old_path (FoundryVcsDelta *delta)
{
  return g_strdup (FOUNDRY_GIT_DELTA (delta)->old_path);
}

static char *
foundry_git_delta_dup_new_path (FoundryVcsDelta *delta)
{
  return g_strdup (FOUNDRY_GIT_DELTA (delta)->new_path);
}

static void
foundry_git_delta_finalize (GObject *object)
{
  FoundryGitDelta *self = (FoundryGitDelta *)object;

  g_clear_pointer (&self->old_path, g_free);
  g_clear_pointer (&self->new_path, g_free);

  G_OBJECT_CLASS (foundry_git_delta_parent_class)->finalize (object);
}

static void
foundry_git_delta_class_init (FoundryGitDeltaClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsDeltaClass *vcs_delta_class = FOUNDRY_VCS_DELTA_CLASS (klass);

  object_class->finalize = foundry_git_delta_finalize;

  vcs_delta_class->dup_old_path = foundry_git_delta_dup_old_path;
  vcs_delta_class->dup_new_path = foundry_git_delta_dup_new_path;
}

static void
foundry_git_delta_init (FoundryGitDelta *self)
{
}

FoundryGitDelta *
_foundry_git_delta_new (const git_diff_delta *delta)
{
  FoundryGitDelta *self;

  g_return_val_if_fail (delta != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_DELTA, NULL);
  self->old_path = g_strdup (delta->old_file.path);
  self->new_path = g_strdup (delta->new_file.path);
  self->old_oid = delta->old_file.id;
  self->new_oid = delta->new_file.id;

  return self;
}
