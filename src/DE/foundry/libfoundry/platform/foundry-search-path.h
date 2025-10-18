/* foundry-search-path.h
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

#pragma once

#include <glib.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

FOUNDRY_AVAILABLE_IN_ALL
char *foundry_search_path_prepend (const char *path,
                                   const char *prepend);
FOUNDRY_AVAILABLE_IN_ALL
char *foundry_search_path_append  (const char *path,
                                   const char *append);

G_END_DECLS
