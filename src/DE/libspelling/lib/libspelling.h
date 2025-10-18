/* libspelling.h
 *
 * Copyright 2023 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <gtk/gtk.h>

G_BEGIN_DECLS

#define LIBSPELLING_INSIDE
# include "spelling-checker.h"
# include "spelling-dictionary.h"
# include "spelling-init.h"
# include "spelling-language.h"
# include "spelling-provider.h"
# include "spelling-text-buffer-adapter.h"
# include "spelling-types.h"
# include "spelling-version.h"
# include "spelling-version-macros.h"
#undef LIBSPELLING_INSIDE

G_END_DECLS
