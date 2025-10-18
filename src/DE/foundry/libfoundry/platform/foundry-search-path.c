/* foundry-search-path.c
 *
 * Copyright 2022-2025 Christian Hergert <chergert@redhat.com>
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

#include "foundry-search-path.h"
#include "foundry-util.h"

char *
foundry_search_path_prepend (const char *path,
                             const char *prepend)
{
  if (foundry_str_empty0 (prepend))
    return g_strdup (path);

  if (foundry_str_empty0 (path))
    return g_strdup (prepend);

  return g_strconcat (prepend, G_SEARCHPATH_SEPARATOR_S, path, NULL);
}

char *
foundry_search_path_append (const char *path,
                            const char *append)
{
  if (foundry_str_empty0 (append))
    return g_strdup (path);

  if (foundry_str_empty0 (path))
    return g_strdup (append);

  return g_strconcat (path, G_SEARCHPATH_SEPARATOR_S, append, NULL);
}
