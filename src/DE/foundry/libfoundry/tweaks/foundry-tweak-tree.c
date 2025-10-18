/* foundry-tweak-tree.c
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

#include "gtktimsortprivate.h"

#include "foundry-internal-tweak.h"
#include "foundry-tweak.h"
#include "foundry-tweak-info-private.h"
#include "foundry-tweak-path.h"
#include "foundry-tweak-tree.h"
#include "foundry-util.h"

typedef struct _Registration
{
  guint              registration;
  FoundryTweakPath  *path;
  FoundryTweakPath  *parent;
  const char        *gettext_domain;
  FoundryTweakInfo **infos;
  guint              n_infos;
} Registration;

struct _FoundryTweakTree
{
  GObject  parent_instance;
  GMutex   mutex;
  GArray  *registrations;
  guint    last_seq;
};

G_DEFINE_FINAL_TYPE (FoundryTweakTree, foundry_tweak_tree, G_TYPE_OBJECT)

static void
clear_registration (Registration *registration)
{
  g_clear_pointer (&registration->path, foundry_tweak_path_free);
  g_clear_pointer (&registration->parent, foundry_tweak_path_free);
  for (guint i = 0; i < registration->n_infos; i++)
    g_clear_pointer (&registration->infos[i], foundry_tweak_info_unref);
  g_clear_pointer (&registration->infos, g_free);
}

static void
foundry_tweak_tree_finalize (GObject *object)
{
  FoundryTweakTree *self = (FoundryTweakTree *)object;

  g_mutex_clear (&self->mutex);
  g_clear_pointer (&self->registrations, g_array_unref);

  G_OBJECT_CLASS (foundry_tweak_tree_parent_class)->finalize (object);
}

static void
foundry_tweak_tree_class_init (FoundryTweakTreeClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_tweak_tree_finalize;
}

static void
foundry_tweak_tree_init (FoundryTweakTree *self)
{
  g_mutex_init (&self->mutex);

  self->registrations = g_array_new (FALSE, FALSE, sizeof (Registration));
  g_array_set_clear_func (self->registrations, (GDestroyNotify)clear_registration);
}

FoundryTweakTree *
foundry_tweak_tree_new (FoundryContext *context)
{
  return g_object_new (FOUNDRY_TYPE_TWEAK_TREE, NULL);
}

static int
sort_by_path (gconstpointer a,
              gconstpointer b)
{
  const Registration *reg_a = a;
  const Registration *reg_b = b;

  return foundry_tweak_path_compare (reg_a->path, reg_b->path);
}

guint
foundry_tweak_tree_register (FoundryTweakTree       *self,
                             const char             *gettext_domain,
                             const char             *path,
                             const FoundryTweakInfo *infos,
                             guint                   n_infos,
                             const char * const     *environment)
{
  g_autoptr(GMutexLocker) locker = NULL;
  Registration reg = {0};

  g_return_val_if_fail (FOUNDRY_IS_TWEAK_TREE (self), 0);
  g_return_val_if_fail (path != NULL, 0);
  g_return_val_if_fail (infos != NULL || n_infos == 0, 0);

  if (n_infos == 0)
    return 0;

  locker = g_mutex_locker_new (&self->mutex);

  reg.registration = ++self->last_seq;
  reg.gettext_domain = g_intern_string (gettext_domain);
  reg.path = foundry_tweak_path_new (path);
  reg.parent = foundry_tweak_path_pop (reg.path);
  reg.infos = g_new0 (FoundryTweakInfo *, n_infos);
  reg.n_infos = n_infos;

  for (guint i = 0; i < n_infos; i++)
    reg.infos[i] = foundry_tweak_info_expand (&infos[i], environment);

  g_array_append_val (self->registrations, reg);
  g_array_sort (self->registrations, sort_by_path);

  return reg.registration;
}

void
foundry_tweak_tree_unregister (FoundryTweakTree *self,
                               guint             registration)
{
  g_autoptr(GMutexLocker) locker = NULL;

  g_return_if_fail (FOUNDRY_IS_TWEAK_TREE (self));
  g_return_if_fail (registration != 0);

  locker = g_mutex_locker_new (&self->mutex);

  for (guint i = 0; i < self->registrations->len; i++)
    {
      const Registration *reg = &g_array_index (self->registrations, Registration, i);

      if (reg->registration == registration)
        {
          g_array_remove_index (self->registrations, i);
          break;
        }
    }
}

static int
foundry_tweak_compare (gconstpointer a,
                       gconstpointer b,
                       gpointer      data)
{
  FoundryTweak *tweak_a = *(FoundryTweak **)a;
  FoundryTweak *tweak_b = *(FoundryTweak **)b;
  g_autofree char *key_a = foundry_tweak_dup_sort_key (tweak_a);
  g_autofree char *key_b = foundry_tweak_dup_sort_key (tweak_b);

  if (key_a == key_b)
    return 0;

  if (key_a == NULL)
    return 1;

  if (key_b == NULL)
    return -1;

  return strcmp (key_a, key_b);
}

static DexFuture *
foundry_tweak_tree_list_fiber (FoundryTweakTree *self,
                               const char       *path)
{
  g_autoptr(GMutexLocker) locker = NULL;
  g_autoptr(FoundryTweakPath) real_path = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GPtrArray) items = NULL;

  g_assert (FOUNDRY_IS_TWEAK_TREE (self));
  g_assert (path != NULL);

  locker = g_mutex_locker_new (&self->mutex);

  real_path = foundry_tweak_path_new (path);
  store = g_list_store_new (G_TYPE_OBJECT);
  items = g_ptr_array_new_with_free_func (g_object_unref);

  for (guint i = 0; i < self->registrations->len; i++)
    {
      const Registration *reg = &g_array_index (self->registrations, Registration, i);
      const FoundryTweakPath *parent = reg->parent;

      if (!foundry_tweak_path_has_prefix (real_path, parent) &&
          !foundry_tweak_path_equal (real_path, parent))
        continue;

      for (guint j = 0; j < reg->n_infos; j++)
        {
          FoundryTweakInfo *info = reg->infos[j];
          g_autoptr(FoundryTweakPath) info_path = foundry_tweak_path_push (reg->path, info->subpath);
          int info_depth = foundry_tweak_path_compute_depth (real_path, info_path);
          g_autoptr(FoundryTweak) tweak = NULL;

          if (info_depth != 1)
            continue;

          tweak = foundry_internal_tweak_new (reg->gettext_domain,
                                              foundry_tweak_info_ref (info),
                                              foundry_tweak_path_dup_path (info_path));

          if (tweak != NULL)
            g_ptr_array_add (items, g_steal_pointer (&tweak));
        }
    }

  if (items->len > 0)
    {
      gtk_tim_sort (items->pdata,
                    items->len,
                    sizeof (gpointer),
                    foundry_tweak_compare,
                    NULL);
      g_list_store_splice (store, 0, 0, items->pdata, items->len);
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

/**
 * foundry_tweak_tree_list:
 * @self: a [class@Foundry.TweakTree]
 * @path:
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.Tweak].
 */
DexFuture *
foundry_tweak_tree_list (FoundryTweakTree *self,
                         const char       *path)
{
  dex_return_error_if_fail (FOUNDRY_IS_TWEAK_TREE (self));
  dex_return_error_if_fail (path != NULL);

  return foundry_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                  G_CALLBACK (foundry_tweak_tree_list_fiber),
                                  2,
                                  FOUNDRY_TYPE_TWEAK_TREE, self,
                                  G_TYPE_STRING, path);
}
