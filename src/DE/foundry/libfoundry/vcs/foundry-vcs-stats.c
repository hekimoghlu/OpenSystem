/* foundry-vcs-stats.c
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

#include "foundry-vcs-stats.h"

enum {
  PROP_0,
  PROP_FILES_CHANGED,
  PROP_INSERTIONS,
  PROP_DELETIONS,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryVcsStats, foundry_vcs_stats, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_vcs_stats_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryVcsStats *self = FOUNDRY_VCS_STATS (object);

  switch (prop_id)
    {
    case PROP_FILES_CHANGED:
      g_value_set_uint64 (value, foundry_vcs_stats_get_files_changed (self));
      break;

    case PROP_INSERTIONS:
      g_value_set_uint64 (value, foundry_vcs_stats_get_insertions (self));
      break;

    case PROP_DELETIONS:
      g_value_set_uint64 (value, foundry_vcs_stats_get_deletions (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_vcs_stats_class_init (FoundryVcsStatsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_vcs_stats_get_property;

  properties[PROP_FILES_CHANGED] =
    g_param_spec_uint64 ("files-changed", NULL, NULL,
                         0, G_MAXUINT64, 0,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_INSERTIONS] =
    g_param_spec_uint64 ("insertions", NULL, NULL,
                         0, G_MAXUINT64, 0,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DELETIONS] =
    g_param_spec_uint64 ("deletions", NULL, NULL,
                         0, G_MAXUINT64, 0,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_vcs_stats_init (FoundryVcsStats *self)
{
}

guint64
foundry_vcs_stats_get_files_changed (FoundryVcsStats *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_STATS (self), 0);

  if (FOUNDRY_VCS_STATS_GET_CLASS (self)->get_files_changed)
    return FOUNDRY_VCS_STATS_GET_CLASS (self)->get_files_changed (self);

  return 0;
}

guint64
foundry_vcs_stats_get_insertions (FoundryVcsStats *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_STATS (self), 0);

  if (FOUNDRY_VCS_STATS_GET_CLASS (self)->get_insertions)
    return FOUNDRY_VCS_STATS_GET_CLASS (self)->get_insertions (self);

  return 0;
}

guint64
foundry_vcs_stats_get_deletions (FoundryVcsStats *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS_STATS (self), 0);

  if (FOUNDRY_VCS_STATS_GET_CLASS (self)->get_deletions)
    return FOUNDRY_VCS_STATS_GET_CLASS (self)->get_deletions (self);

  return 0;
}
