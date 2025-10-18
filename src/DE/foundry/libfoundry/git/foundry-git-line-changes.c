/* foundry-git-line-changes.c
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

#include "foundry-git-line-changes-private.h"

struct _FoundryGitLineChanges
{
  FoundryVcsLineChanges parent_instance;
  LineCache *cache;
};

G_DEFINE_FINAL_TYPE (FoundryGitLineChanges, foundry_git_line_changes, FOUNDRY_TYPE_VCS_LINE_CHANGES)

static FoundryVcsLineChange
foundry_git_line_changes_query_line (FoundryVcsLineChanges *changes,
                                     guint                  line)
{
  FoundryGitLineChanges *self = (FoundryGitLineChanges *)changes;

  return (FoundryVcsLineChange)line_cache_get_mark (self->cache, line);
}

typedef struct
{
  FoundryVcsLineChangesForeach callback;
  gpointer user_data;
} Wrapper;

static void
wrapper_foreach (gpointer data,
                 gpointer user_data)
{
  const LineEntry *entry = data;
  Wrapper *state = user_data;

  state->callback (entry->line, entry->mark, state->user_data);
}

static void
foundry_git_line_changes_foreach (FoundryVcsLineChanges        *changes,
                                  guint                         first_line,
                                  guint                         last_line,
                                  FoundryVcsLineChangesForeach  foreach,
                                  gpointer                      user_data)
{
  FoundryGitLineChanges *self = (FoundryGitLineChanges *)changes;
  Wrapper state;

  g_assert (FOUNDRY_IS_GIT_LINE_CHANGES (self));
  g_assert (foreach != NULL);

  state.callback = foreach;
  state.user_data = user_data;

  line_cache_foreach_in_range (self->cache, first_line, last_line, wrapper_foreach, &state);
}

static void
foundry_git_line_changes_finalize (GObject *object)
{
  FoundryGitLineChanges *self = (FoundryGitLineChanges *)object;

  g_clear_pointer (&self->cache, line_cache_free);

  G_OBJECT_CLASS (foundry_git_line_changes_parent_class)->finalize (object);
}

static void
foundry_git_line_changes_class_init (FoundryGitLineChangesClass *klass)
{
  FoundryVcsLineChangesClass *changes_class = FOUNDRY_VCS_LINE_CHANGES_CLASS (klass);
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_git_line_changes_finalize;

  changes_class->query_line = foundry_git_line_changes_query_line;
  changes_class->foreach = foundry_git_line_changes_foreach;
}

static void
foundry_git_line_changes_init (FoundryGitLineChanges *self)
{
}

FoundryVcsLineChanges *
_foundry_git_line_changes_new (LineCache *cache)
{
  FoundryGitLineChanges *self;

  g_return_val_if_fail (cache != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_LINE_CHANGES, NULL);
  self->cache = g_steal_pointer (&cache);

  return FOUNDRY_VCS_LINE_CHANGES (self);
}
