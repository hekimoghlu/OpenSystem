/* foundry-vcs-tag.h
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

#pragma once

#include "foundry-vcs-reference.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_VCS_TAG (foundry_vcs_tag_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryVcsTag, foundry_vcs_tag, FOUNDRY, VCS_TAG, GObject)

struct _FoundryVcsTagClass
{
  GObjectClass parent_class;

  char      *(*dup_id)      (FoundryVcsTag *self);
  char      *(*dup_title)   (FoundryVcsTag *self);
  gboolean   (*is_local)    (FoundryVcsTag *self);
  DexFuture *(*load_target) (FoundryVcsTag *self);

  /*< private >*/
  gpointer _reserved[19];
};

FOUNDRY_AVAILABLE_IN_ALL
gboolean   foundry_vcs_tag_is_local    (FoundryVcsTag *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_tag_dup_id      (FoundryVcsTag *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_tag_dup_title   (FoundryVcsTag *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture *foundry_vcs_tag_load_target (FoundryVcsTag *self);

G_END_DECLS
