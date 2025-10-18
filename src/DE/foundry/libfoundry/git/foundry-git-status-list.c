/* foundry-git-status-list.c
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
#include <git2.h>

#include "foundry-git-status-entry-private.h"
#include "foundry-git-status-list-private.h"

static void
maybe_free (gpointer data)
{
  if (data)
    g_object_unref (data);
}

#define EGG_ARRAY_NAME items
#define EGG_ARRAY_TYPE_NAME Items
#define EGG_ARRAY_ELEMENT_TYPE FoundryGitStatusEntry*
#define EGG_ARRAY_FREE_FUNC maybe_free
#include "eggarrayimpl.c"

struct _FoundryGitStatusList
{
  GObject          parent_instance;
  git_status_list *status_list;
  Items            items;
};

static GType
foundry_git_status_list_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_GIT_STATUS_ENTRY;
}

static guint
foundry_git_status_list_get_n_items (GListModel *model)
{
  FoundryGitStatusList *self = FOUNDRY_GIT_STATUS_LIST (model);

  return git_status_list_entrycount (self->status_list);
}

static gpointer
foundry_git_status_list_get_item (GListModel *model,
                                  guint       position)
{
  FoundryGitStatusList *self = FOUNDRY_GIT_STATUS_LIST (model);
  FoundryGitStatusEntry *entry;

  /* Some GTK list models expect that our objects are persistent
   * after fetching until they are invalidated or last reference
   * has been dropped.
   *
   * That gives us two options:
   *
   *   - Retain all the objects after creating (which we do)
   *   - Weak ref track all of them to cleanup
   *
   * Currently we opt for the first simply out of the goal
   * to avoid the expensive runtime costs of weak pointers,
   * though we may try to avoid that in the future with
   * backpointers to be cleared instead.
   */

  if (position >= items_get_size (&self->items))
    return NULL;

  if (!(entry = items_get (&self->items, position)))
    {
      const git_status_entry *e = git_status_byindex (self->status_list, position);

      entry = _foundry_git_status_entry_new (e);
      self->items.start[position] = entry;
    }

  return g_object_ref (entry);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_git_status_list_get_item_type;
  iface->get_n_items = foundry_git_status_list_get_n_items;
  iface->get_item = foundry_git_status_list_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryGitStatusList, foundry_git_status_list, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static void
foundry_git_status_list_finalize (GObject *object)
{
  FoundryGitStatusList *self = (FoundryGitStatusList *)object;

  items_clear (&self->items);
  g_clear_pointer (&self->status_list, git_status_list_free);

  G_OBJECT_CLASS (foundry_git_status_list_parent_class)->finalize (object);
}

static void
foundry_git_status_list_class_init (FoundryGitStatusListClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_git_status_list_finalize;
}

static void
foundry_git_status_list_init (FoundryGitStatusList *self)
{
  items_init (&self->items);
}

FoundryGitStatusList *
_foundry_git_status_list_new (git_status_list *status_list)
{
  FoundryGitStatusList *self;

  g_return_val_if_fail (status_list != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_STATUS_LIST, NULL);
  self->status_list = status_list;
  items_set_size (&self->items, git_status_list_entrycount (status_list));

  return self;
}
