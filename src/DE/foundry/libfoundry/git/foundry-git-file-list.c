/* foundry-git-file-list.c
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

#include "foundry-git-file-list-private.h"
#include "foundry-git-file-private.h"

static void
maybe_free (gpointer data)
{
  if (data)
    g_object_unref (data);
}

#define EGG_ARRAY_NAME items
#define EGG_ARRAY_TYPE_NAME Items
#define EGG_ARRAY_ELEMENT_TYPE FoundryGitFile*
#define EGG_ARRAY_FREE_FUNC maybe_free
#include "eggarrayimpl.c"

struct _FoundryGitFileList
{
  GObject    parent_instance;
  git_index *index;
  GFile     *workdir;
  Items      items;
};

static GType
foundry_git_file_list_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_VCS_FILE;
}

static guint
foundry_git_file_list_get_n_items (GListModel *model)
{
  FoundryGitFileList *self = FOUNDRY_GIT_FILE_LIST (model);

  return git_index_entrycount (self->index);
}

static gpointer
foundry_git_file_list_get_item (GListModel *model,
                                guint       position)
{
  FoundryGitFileList *self = FOUNDRY_GIT_FILE_LIST (model);
  FoundryGitFile *file;

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

  if (!(file = items_get (&self->items, position)))
    {
      const git_index_entry *entry;

      if (position >= git_index_entrycount (self->index))
        return NULL;

      if (!(entry = git_index_get_byindex (self->index, position)))
        return NULL;

      file = _foundry_git_file_new (self->workdir, entry->path);
      self->items.start[position] = file;
    }

  return g_object_ref (file);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_git_file_list_get_item_type;
  iface->get_n_items = foundry_git_file_list_get_n_items;
  iface->get_item = foundry_git_file_list_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryGitFileList, foundry_git_file_list, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static void
foundry_git_file_list_finalize (GObject *object)
{
  FoundryGitFileList *self = (FoundryGitFileList *)object;

  items_clear (&self->items);
  g_clear_pointer (&self->index, git_index_free);
  g_clear_object (&self->workdir);

  G_OBJECT_CLASS (foundry_git_file_list_parent_class)->finalize (object);
}

static void
foundry_git_file_list_class_init (FoundryGitFileListClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_git_file_list_finalize;
}

static void
foundry_git_file_list_init (FoundryGitFileList *self)
{
  items_init (&self->items);
}

FoundryGitFileList *
_foundry_git_file_list_new (GFile     *workdir,
                            git_index *index)
{
  FoundryGitFileList *self;

  g_return_val_if_fail (G_IS_FILE (workdir), NULL);
  g_return_val_if_fail (index != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_FILE_LIST, NULL);
  self->workdir = g_object_ref (workdir);
  self->index = g_steal_pointer (&index);

  items_set_size (&self->items, git_index_entrycount (self->index));

  return self;
}
