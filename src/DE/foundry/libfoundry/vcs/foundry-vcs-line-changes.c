/* foundry-vcs-line-changes.c
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

#include "foundry-vcs-line-changes.h"

G_DEFINE_ABSTRACT_TYPE (FoundryVcsLineChanges, foundry_vcs_line_changes, G_TYPE_OBJECT)

static void
foundry_vcs_line_changes_class_init (FoundryVcsLineChangesClass *klass)
{
}

static void
foundry_vcs_line_changes_init (FoundryVcsLineChanges *self)
{
}

FoundryVcsLineChange
foundry_vcs_line_changes_query_line (FoundryVcsLineChanges *self,
                                     guint                  line)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_LINE_CHANGES (self), 0);

  if (FOUNDRY_VCS_LINE_CHANGES_GET_CLASS (self)->query_line)
    return FOUNDRY_VCS_LINE_CHANGES_GET_CLASS (self)->query_line (self, line);

  return 0;
}

/**
 * foundry_vcs_line_changes_foreach:
 * @self: a [class@Foundry.VcsLineChanges]
 * @foreach: (scope call):
 *
 */
void
foundry_vcs_line_changes_foreach (FoundryVcsLineChanges        *self,
                                  guint                         first_line,
                                  guint                         last_line,
                                  FoundryVcsLineChangesForeach  foreach,
                                  gpointer                      user_data)
{
  g_return_if_fail (FOUNDRY_IS_VCS_LINE_CHANGES (self));
  g_return_if_fail (foreach != NULL);

  if (first_line > last_line)
    {
      guint tmp = first_line;
      first_line = last_line;
      last_line = tmp;
    }

  last_line = MIN (G_MAXUINT - 1, last_line);

  if (FOUNDRY_VCS_LINE_CHANGES_GET_CLASS (self)->foreach)
    {
      FOUNDRY_VCS_LINE_CHANGES_GET_CLASS (self)->foreach (self, first_line, last_line, foreach, user_data);
    }
  else
    {
      for (guint i = first_line; i <= last_line; i++)
        {
          FoundryVcsLineChange change = foundry_vcs_line_changes_query_line (self, i);

          if (change != 0)
            foreach (i, change, user_data);
        }
    }
}

G_DEFINE_FLAGS_TYPE (FoundryVcsLineChange, foundry_vcs_line_change,
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_LINE_ADDED, "added"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_LINE_REMOVED, "removed"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_LINE_CHANGED, "changed"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_LINE_PREVIOUS_REMOVED, "previous-removed"))
