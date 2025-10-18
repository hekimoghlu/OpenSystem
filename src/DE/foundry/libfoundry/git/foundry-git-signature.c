/* foundry-git-signature.c
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

#include "foundry-git-signature-private.h"
#include "foundry-git-time.h"

struct _FoundryGitSignature
{
  FoundryVcsSignature  parent_instance;
  git_signature       *signature;
};

G_DEFINE_FINAL_TYPE (FoundryGitSignature, foundry_git_signature, FOUNDRY_TYPE_VCS_SIGNATURE)

static char *
foundry_git_signature_dup_name (FoundryVcsSignature *signature)
{
  FoundryGitSignature *self = (FoundryGitSignature *)signature;

  g_assert (FOUNDRY_IS_GIT_SIGNATURE (self));

  if (self->signature->name != NULL)
    return g_utf8_make_valid (self->signature->name, -1);

  return NULL;
}

static char *
foundry_git_signature_dup_email (FoundryVcsSignature *signature)
{
  FoundryGitSignature *self = (FoundryGitSignature *)signature;

  g_assert (FOUNDRY_IS_GIT_SIGNATURE (self));

  if (self->signature->email != NULL)
    return g_utf8_make_valid (self->signature->email, -1);

  return NULL;
}

static GDateTime *
foundry_git_signature_dup_when (FoundryVcsSignature *signature)
{
  FoundryGitSignature *self = (FoundryGitSignature *)signature;
  g_autoptr(GMutexLocker) locker = NULL;

  g_assert (FOUNDRY_IS_GIT_SIGNATURE (self));

  return foundry_git_time_to_date_time (&self->signature->when);
}

static void
foundry_git_signature_finalize (GObject *object)
{
  FoundryGitSignature *self = (FoundryGitSignature *)object;

  g_clear_pointer (&self->signature, git_signature_free);

  G_OBJECT_CLASS (foundry_git_signature_parent_class)->finalize (object);
}

static void
foundry_git_signature_class_init (FoundryGitSignatureClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsSignatureClass *signature_class = FOUNDRY_VCS_SIGNATURE_CLASS (klass);

  object_class->finalize = foundry_git_signature_finalize;

  signature_class->dup_name = foundry_git_signature_dup_name;
  signature_class->dup_email = foundry_git_signature_dup_email;
  signature_class->dup_when = foundry_git_signature_dup_when;
}

static void
foundry_git_signature_init (FoundryGitSignature *self)
{
}

FoundryVcsSignature *
_foundry_git_signature_new (git_signature *signature)
{
  FoundryGitSignature *self;

  g_return_val_if_fail (signature != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_SIGNATURE, NULL);
  self->signature = g_steal_pointer (&signature);

  return FOUNDRY_VCS_SIGNATURE (self);
}
