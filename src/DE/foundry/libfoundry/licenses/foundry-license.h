/* foundry-license.h
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

#include <gio/gio.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LICENSE (foundry_license_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryLicense, foundry_license, FOUNDRY, LICENSE, GObject)

FOUNDRY_AVAILABLE_IN_ALL
GBytes         *foundry_license_dup_text         (FoundryLicense *self);
FOUNDRY_AVAILABLE_IN_ALL
GBytes         *foundry_license_dup_snippet_text (FoundryLicense *self);
FOUNDRY_AVAILABLE_IN_ALL
char           *foundry_license_dup_id           (FoundryLicense *self);
FOUNDRY_AVAILABLE_IN_ALL
char           *foundry_license_dup_title        (FoundryLicense *self);
FOUNDRY_AVAILABLE_IN_ALL
GListModel     *foundry_license_list_all         (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryLicense *foundry_license_find             (const char     *id);

G_END_DECLS
