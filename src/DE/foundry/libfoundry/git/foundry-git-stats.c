/* foundry-git-stats.c
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

#include "foundry-git-stats-private.h"

struct _FoundryGitStats
{
  FoundryVcsStats parent_instance;
  guint64 files_changed;
  guint64 insertions;
  guint64 deletions;
};

G_DEFINE_FINAL_TYPE (FoundryGitStats, foundry_git_stats, FOUNDRY_TYPE_VCS_STATS)

static guint64
foundry_git_stats_get_files_changed (FoundryVcsStats *stats)
{
  return FOUNDRY_GIT_STATS (stats)->files_changed;
}

static guint64
foundry_git_stats_get_insertions (FoundryVcsStats *stats)
{
  return FOUNDRY_GIT_STATS (stats)->insertions;
}

static guint64
foundry_git_stats_get_deletions (FoundryVcsStats *stats)
{
  return FOUNDRY_GIT_STATS (stats)->deletions;
}

static void
foundry_git_stats_class_init (FoundryGitStatsClass *klass)
{
  FoundryVcsStatsClass *stats_class = FOUNDRY_VCS_STATS_CLASS (klass);

  stats_class->get_files_changed = foundry_git_stats_get_files_changed;
  stats_class->get_insertions = foundry_git_stats_get_insertions;
  stats_class->get_deletions = foundry_git_stats_get_deletions;
}

static void
foundry_git_stats_init (FoundryGitStats *self)
{
}

FoundryGitStats *
_foundry_git_stats_new (const git_diff_stats *stats)
{
  FoundryGitStats *self;

  g_return_val_if_fail (stats != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_STATS, NULL);
  self->files_changed = git_diff_stats_files_changed (stats);
  self->insertions = git_diff_stats_insertions (stats);
  self->deletions = git_diff_stats_deletions (stats);

  return self;
}
