/* foundry-shortcut-manager.h
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

#include <foundry.h>
#include <libpeas.h>

#include "foundry-shortcut-bundle.h"
#include "foundry-shortcut-observer.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SHORTCUT_MANAGER (foundry_shortcut_manager_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryShortcutManager, foundry_shortcut_manager, FOUNDRY, SHORTCUT_MANAGER, FoundryService)

FOUNDRY_AVAILABLE_IN_ALL
FoundryShortcutManager  *foundry_shortcut_manager_from_context     (FoundryContext         *context);
FOUNDRY_AVAILABLE_IN_ALL
void                     foundry_shortcut_manager_add_resources    (const char             *resource_path);
FOUNDRY_AVAILABLE_IN_ALL
void                     foundry_shortcut_manager_remove_resources (const char             *resource_path);
FOUNDRY_AVAILABLE_IN_ALL
FoundryShortcutObserver *foundry_shortcut_manager_get_observer     (FoundryShortcutManager *self);
FOUNDRY_AVAILABLE_IN_ALL
void                     foundry_shortcut_manager_reset_user       (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryShortcutBundle   *foundry_shortcut_manager_get_user_bundle  (void);

G_END_DECLS
