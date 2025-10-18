/* foundry-flatpak-sources.c
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
#include "foundry-flatpak-source.h"
#include "foundry-flatpak-source-archive.h"
#include "foundry-flatpak-source-bzr.h"
#include "foundry-flatpak-source-dir.h"
#include "foundry-flatpak-source-extra-data.h"
#include "foundry-flatpak-source-file.h"
#include "foundry-flatpak-source-git.h"
#include "foundry-flatpak-source-inline.h"
#include "foundry-flatpak-source-patch.h"
#include "foundry-flatpak-source-script.h"
#include "foundry-flatpak-source-shell.h"
#include "foundry-flatpak-source-svn.h"
#include "foundry-flatpak-sources.h"

struct _FoundryFlatpakSources
{
  FoundryFlatpakList parent_instance;
};

struct _FoundryFlatpakSourcesClass
{
  FoundryFlatpakListClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakSources, foundry_flatpak_sources, FOUNDRY_TYPE_FLATPAK_LIST)

static GType
foundry_flatpak_sources_get_item_type (FoundryFlatpakList *self,
                                       const char         *type)
{
  g_autofree GType *children = NULL;
  guint n_children = 0;

  if ((children = g_type_children (FOUNDRY_TYPE_FLATPAK_SOURCE, &n_children)))
    {
      for (guint i = 0; i < n_children; i++)
        {
          g_autoptr(FoundryFlatpakSourceClass) klass = NULL;

          if (G_TYPE_IS_ABSTRACT (children[i]))
            continue;

          klass = g_type_class_ref (children[i]);

          if (g_strcmp0 (type, klass->type) == 0)
            return children[i];
        }
    }

  return FOUNDRY_FLATPAK_LIST_GET_CLASS (self)->item_type;
}

static void
foundry_flatpak_sources_class_init (FoundryFlatpakSourcesClass *klass)
{
  FoundryFlatpakListClass *list_class = FOUNDRY_FLATPAK_LIST_CLASS (klass);

  list_class->item_type = FOUNDRY_TYPE_FLATPAK_SOURCE;
  list_class->get_item_type = foundry_flatpak_sources_get_item_type;

  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_ARCHIVE);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_BZR);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_DIR);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_EXTRA_DATA);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_FILE);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_GIT);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_INLINE);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_PATCH);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_SCRIPT);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_SHELL);
  g_type_ensure (FOUNDRY_TYPE_FLATPAK_SOURCE_SVN);
}

static void
foundry_flatpak_sources_init (FoundryFlatpakSources *self)
{
}
