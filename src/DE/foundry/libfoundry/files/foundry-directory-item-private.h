/* foundry-directory-item-private.h
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

#pragma once

#include "foundry-directory-item.h"

G_BEGIN_DECLS

struct _FoundryDirectoryItem
{
  GObject        parent_instance;
  GSequenceIter *iter;
  GFile         *directory;
  GFile         *file;
  GFileInfo     *info;
};

static inline FoundryDirectoryItem *
foundry_directory_item_new (GFile     *directory,
                            GFile     *file,
                            GFileInfo *info)
{
  g_assert (G_IS_FILE (directory));
  g_assert (G_IS_FILE (file));
  g_assert (G_IS_FILE_INFO (info));

  return g_object_new (FOUNDRY_TYPE_DIRECTORY_ITEM,
                       "directory", directory,
                       "file", file,
                       "info", info,
                       NULL);
}

G_END_DECLS
