/* foundry-vcs-blame.h
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

#include <libdex.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_VCS_BLAME (foundry_vcs_blame_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryVcsBlame, foundry_vcs_blame, FOUNDRY, VCS_BLAME, GObject)

struct _FoundryVcsBlameClass
{
  GObjectClass parent_class;

  DexFuture           *(*update)      (FoundryVcsBlame *self,
                                       GBytes          *bytes);
  guint                (*get_n_lines) (FoundryVcsBlame *self);
  FoundryVcsSignature *(*query_line)  (FoundryVcsBlame *self,
                                       guint            line);

  /*< private >*/
  gpointer _reserved[12];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryVcsFile      *foundry_vcs_blame_dup_file    (FoundryVcsBlame *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture           *foundry_vcs_blame_update      (FoundryVcsBlame *self,
                                                    GBytes          *bytes);
FOUNDRY_AVAILABLE_IN_ALL
FoundryVcsSignature *foundry_vcs_blame_query_line  (FoundryVcsBlame *self,
                                                    guint            line);
FOUNDRY_AVAILABLE_IN_ALL
guint                foundry_vcs_blame_get_n_lines (FoundryVcsBlame *self);

G_END_DECLS
