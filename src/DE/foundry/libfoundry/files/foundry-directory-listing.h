/* foundry-directory-listing.h
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

#include <libdex.h>

#include "foundry-contextual.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DIRECTORY_LISTING (foundry_directory_listing_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryDirectoryListing, foundry_directory_listing, FOUNDRY, DIRECTORY_LISTING, FoundryContextual)

FOUNDRY_AVAILABLE_IN_ALL
FoundryDirectoryListing *foundry_directory_listing_new   (FoundryContext          *context,
                                                          GFile                   *directory,
                                                          const char              *attributes,
                                                          GFileQueryInfoFlags      query_flags);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture               *foundry_directory_listing_await (FoundryDirectoryListing *self);

G_END_DECLS
