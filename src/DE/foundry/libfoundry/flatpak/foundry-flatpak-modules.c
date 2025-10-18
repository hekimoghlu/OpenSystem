/* foundry-flatpak-modules.c
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

#include "foundry-flatpak-list-private.h"
#include "foundry-flatpak-module.h"
#include "foundry-flatpak-modules.h"

struct _FoundryFlatpakModules
{
  FoundryFlatpakList parent_instance;
};

struct _FoundryFlatpakModulesClass
{
  FoundryFlatpakListClass parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakModules, foundry_flatpak_modules, FOUNDRY_TYPE_FLATPAK_LIST)

static void
foundry_flatpak_modules_class_init (FoundryFlatpakModulesClass *klass)
{
  FoundryFlatpakListClass *list_class = FOUNDRY_FLATPAK_LIST_CLASS (klass);

  list_class->item_type = FOUNDRY_TYPE_FLATPAK_MODULE;
}

static void
foundry_flatpak_modules_init (FoundryFlatpakModules *self)
{
}

/**
 * foundry_flatpak_modules_find_primary:
 * @self: a [class@Foundry.FlatpakModules]
 * @project_dir: the directory of the project
 *
 * Returns: (transfer full) (nullable):
 */
FoundryFlatpakModule *
foundry_flatpak_modules_find_primary (FoundryFlatpakModules *self,
                                      GFile                 *project_dir)
{
  g_autofree char *project_basename = NULL;
  guint n_items;

  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MODULES (self), NULL);
  g_return_val_if_fail (G_IS_FILE (project_dir), NULL);

  project_basename = g_file_get_basename (project_dir);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryFlatpakModule) module = g_list_model_get_item (G_LIST_MODEL (self), i);
      g_autoptr(FoundryFlatpakModule) submodule = NULL;
      g_autoptr(FoundryFlatpakModules) modules = NULL;
      g_autofree char *name = foundry_flatpak_module_dup_name (module);

      if (g_strcmp0 (project_basename, name) == 0)
        return g_steal_pointer (&module);

      if ((modules = foundry_flatpak_module_dup_modules (module)))
        {
          if ((submodule = foundry_flatpak_modules_find_primary (modules, project_dir)))
            return g_steal_pointer (&submodule);
        }
    }

  return NULL;
}
