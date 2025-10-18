/* foundry-no-vcs.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-no-vcs.h"

struct _FoundryNoVcs
{
  FoundryVcs parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundryNoVcs, foundry_no_vcs, FOUNDRY_TYPE_VCS)

static char *
foundry_no_vcs_dup_id (FoundryVcs *vcs)
{
  return g_strdup ("no");
}

static char *
foundry_no_vcs_dup_name (FoundryVcs *vcs)
{
  return g_strdup (_("No Version Control"));
}

static char *
foundry_no_vcs_dup_branch_name (FoundryVcs *vcs)
{
  return g_strdup ("");
}

static gboolean
foundry_no_vcs_is_file_ignored (FoundryVcs *vcs,
                                GFile      *file)
{
  g_autofree char *base = g_file_get_basename (file);
  return base[0] == '.';
}

static void
foundry_no_vcs_class_init (FoundryNoVcsClass *klass)
{
  FoundryVcsClass *vcs_class = FOUNDRY_VCS_CLASS (klass);

  vcs_class->dup_id = foundry_no_vcs_dup_id;
  vcs_class->dup_name = foundry_no_vcs_dup_name;
  vcs_class->dup_branch_name = foundry_no_vcs_dup_branch_name;
  vcs_class->is_file_ignored = foundry_no_vcs_is_file_ignored;
}

static void
foundry_no_vcs_init (FoundryNoVcs *self)
{
}

/**
 * foundry_no_vcs_new:
 * @context: a [class@Foundry.Context]
 *
 * Creates a new "no-op" VCS.
 *
 * Returns: (transfer full): a [class@Foundry.Vcs].
 */
FoundryVcs *
foundry_no_vcs_new (FoundryContext *context)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  return g_object_new (FOUNDRY_TYPE_NO_VCS,
                       "context", context,
                       NULL);
}
