/* foundry-vcs-diff.c
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

#include "foundry-vcs-diff.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryVcsDiff, foundry_vcs_diff, G_TYPE_OBJECT)

static void
foundry_vcs_diff_class_init (FoundryVcsDiffClass *klass)
{
}

static void
foundry_vcs_diff_init (FoundryVcsDiff *self)
{
}

/**
 * foundry_vcs_diff_list_deltas:
 * @self: a [class@Foundry.VcsDiff]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.VcsDelta] or rejects with error
 */
DexFuture *
foundry_vcs_diff_list_deltas (FoundryVcsDiff *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS_DIFF (self));

  if (FOUNDRY_VCS_DIFF_GET_CLASS (self)->list_deltas)
    return FOUNDRY_VCS_DIFF_GET_CLASS (self)->list_deltas (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_diff_load_stats:
 * @self: a [class@Foundry.VcsDiff]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsStats] or rejects with error
 */
DexFuture *
foundry_vcs_diff_load_stats (FoundryVcsDiff *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS_DIFF (self));

  if (FOUNDRY_VCS_DIFF_GET_CLASS (self)->load_stats)
    return FOUNDRY_VCS_DIFF_GET_CLASS (self)->load_stats (self);

  return foundry_future_new_not_supported ();
}
