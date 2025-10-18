/* foundry-git-status-entry.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <gio/gio.h>

#include "foundry-git-status-entry-private.h"

struct _FoundryGitStatusEntry
{
  GObject parent_instance;
  char *path;
  guint has_staged_changes : 1;
  guint has_unstaged_changes : 1;
  guint is_new_file : 1;
  guint is_modified : 1;
  guint is_removed : 1;
};

G_DEFINE_FINAL_TYPE (FoundryGitStatusEntry, foundry_git_status_entry, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_HAS_STAGED_CHANGES,
  PROP_HAS_UNSTAGED_CHANGES,
  PROP_ICON,
  PROP_IS_NEW_FILE,
  PROP_PATH,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_git_status_entry_finalize (GObject *object)
{
  FoundryGitStatusEntry *self = (FoundryGitStatusEntry *)object;

  g_clear_pointer (&self->path, g_free);

  G_OBJECT_CLASS (foundry_git_status_entry_parent_class)->finalize (object);
}

static void
foundry_git_status_entry_get_property (GObject    *object,
                                       guint       prop_id,
                                       GValue     *value,
                                       GParamSpec *pspec)
{
  FoundryGitStatusEntry *self = FOUNDRY_GIT_STATUS_ENTRY (object);

  switch (prop_id)
    {
    case PROP_HAS_STAGED_CHANGES:
      g_value_set_boolean (value, foundry_git_status_entry_has_staged_changed (self));
      break;

    case PROP_HAS_UNSTAGED_CHANGES:
      g_value_set_boolean (value, foundry_git_status_entry_has_unstaged_changed (self));
      break;

    case PROP_ICON:
      g_value_take_object (value, foundry_git_status_entry_dup_icon (self));
      break;

    case PROP_IS_NEW_FILE:
      g_value_set_boolean (value, foundry_git_status_entry_is_new_file (self));
      break;

    case PROP_PATH:
      g_value_take_string (value, foundry_git_status_entry_dup_path (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_git_status_entry_class_init (FoundryGitStatusEntryClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_git_status_entry_finalize;
  object_class->get_property = foundry_git_status_entry_get_property;

  properties[PROP_PATH] =
    g_param_spec_string ("path", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_HAS_STAGED_CHANGES] =
    g_param_spec_boolean ("has-staged-changes", NULL, NULL,
                         FALSE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_HAS_UNSTAGED_CHANGES] =
    g_param_spec_boolean ("has-unstaged-changes", NULL, NULL,
                         FALSE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_IS_NEW_FILE] =
    g_param_spec_boolean ("is-new-file", NULL, NULL,
                         FALSE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_git_status_entry_init (FoundryGitStatusEntry *self)
{
}

FoundryGitStatusEntry *
_foundry_git_status_entry_new (const git_status_entry *entry)
{
  FoundryGitStatusEntry *self;
  const char *path = NULL;

  g_return_val_if_fail (entry != NULL, NULL);

  if (entry->head_to_index)
    path = entry->head_to_index->new_file.path;
  else if (entry->index_to_workdir)
    path = entry->index_to_workdir->new_file.path;

  if (path == NULL)
    return NULL;

  self = g_object_new (FOUNDRY_TYPE_GIT_STATUS_ENTRY, NULL);
  self->path = g_strdup (path);
  self->has_staged_changes = !!(entry->status & (GIT_STATUS_INDEX_NEW |
                                                 GIT_STATUS_INDEX_MODIFIED |
                                                 GIT_STATUS_INDEX_DELETED |
                                                 GIT_STATUS_INDEX_RENAMED |
                                                 GIT_STATUS_INDEX_TYPECHANGE));
  self->has_unstaged_changes = !!(entry->status & (GIT_STATUS_WT_NEW |
                                                   GIT_STATUS_WT_MODIFIED |
                                                   GIT_STATUS_WT_DELETED |
                                                   GIT_STATUS_WT_RENAMED |
                                                   GIT_STATUS_WT_TYPECHANGE));
  self->is_new_file = !!(entry->status & (GIT_STATUS_WT_NEW | GIT_STATUS_INDEX_NEW));
  self->is_modified = !!(entry->status & (GIT_STATUS_WT_MODIFIED | GIT_STATUS_INDEX_MODIFIED));
  self->is_removed = !!(entry->status & (GIT_STATUS_WT_DELETED | GIT_STATUS_INDEX_DELETED));

  return self;
}

gboolean
foundry_git_status_entry_has_unstaged_changed (FoundryGitStatusEntry *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_STATUS_ENTRY (self), FALSE);

  return self->has_unstaged_changes;
}

gboolean
foundry_git_status_entry_has_staged_changed (FoundryGitStatusEntry *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_STATUS_ENTRY (self), FALSE);

  return self->has_staged_changes;
}

gboolean
foundry_git_status_entry_is_new_file (FoundryGitStatusEntry *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_STATUS_ENTRY (self), FALSE);

  return self->is_new_file;
}

char *
foundry_git_status_entry_dup_path (FoundryGitStatusEntry *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_STATUS_ENTRY (self), NULL);

  return g_strdup (self->path);
}

/**
 * foundry_git_status_entry_dup_icon:
 * @self: a [class@Foundry.GitStatusEntry]
 *
 * Returns: (transfer full) (nullable):
 */
GIcon *
foundry_git_status_entry_dup_icon (FoundryGitStatusEntry *self)
{
  static GIcon *changed_icon;
  static GIcon *removed_icon;
  static GIcon *new_icon;

  g_return_val_if_fail (FOUNDRY_IS_GIT_STATUS_ENTRY (self), NULL);

  if (self->is_new_file)
    {
      if (new_icon == NULL)
        new_icon = g_themed_icon_new ("vcs-file-added-symbolic");

      return g_object_ref (new_icon);
    }

  if (self->is_modified)
    {
      if (changed_icon == NULL)
        changed_icon = g_themed_icon_new ("vcs-file-changed-symbolic");

      return g_object_ref (changed_icon);
    }

  if (self->is_removed)
    {
      if (removed_icon == NULL)
        removed_icon = g_themed_icon_new ("vcs-file-removed-symbolic");

      return g_object_ref (removed_icon);
    }

  return NULL;
}
