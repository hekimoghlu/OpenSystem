/* foundry-vcs-line-changes.h
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

#include <glib-object.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_VCS_LINE_CHANGES (foundry_vcs_line_changes_get_type())
#define FOUNDRY_TYPE_VCS_LINE_CHANGE  (foundry_vcs_line_change_get_type())

typedef enum _FoundryVcsLineChange
{
  FOUNDRY_VCS_LINE_ADDED            = 1 << 0,
  FOUNDRY_VCS_LINE_REMOVED          = 1 << 1,
  FOUNDRY_VCS_LINE_CHANGED          = 1 << 2,
  FOUNDRY_VCS_LINE_PREVIOUS_REMOVED = 1 << 3,
} FoundryVcsLineChange;

typedef void (*FoundryVcsLineChangesForeach) (guint                line,
                                              FoundryVcsLineChange change,
                                              gpointer             user_data);

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryVcsLineChanges, foundry_vcs_line_changes, FOUNDRY, VCS_LINE_CHANGES, GObject)

struct _FoundryVcsLineChangesClass
{
  GObjectClass parent_class;

  FoundryVcsLineChange (*query_line) (FoundryVcsLineChanges        *self,
                                      guint                         line);
  void                 (*foreach)    (FoundryVcsLineChanges        *self,
                                      guint                         first_line,
                                      guint                         last_line,
                                      FoundryVcsLineChangesForeach  foreach,
                                      gpointer                      user_data);

  /*< private >*/
  gpointer _reserved[13];
};

FOUNDRY_AVAILABLE_IN_ALL
GType                  foundry_vcs_line_change_get_type    (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryVcsLineChange   foundry_vcs_line_changes_query_line (FoundryVcsLineChanges        *self,
                                                            guint                         line);
FOUNDRY_AVAILABLE_IN_ALL
void                   foundry_vcs_line_changes_foreach    (FoundryVcsLineChanges        *self,
                                                            guint                         first_line,
                                                            guint                         last_line,
                                                            FoundryVcsLineChangesForeach  foreach,
                                                            gpointer                      user_data);

G_END_DECLS
