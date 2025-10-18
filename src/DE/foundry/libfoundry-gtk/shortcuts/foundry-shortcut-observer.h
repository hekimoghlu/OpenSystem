/* foundry-shortcut-observer.h
 *
 * Copyright 2023-2025 Christian Hergert <chergert@redhat.com>
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

#include <gtk/gtk.h>

#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_SHORTCUT_OBSERVER (foundry_shortcut_observer_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryShortcutObserver, foundry_shortcut_observer, FOUNDRY, SHORTCUT_OBSERVER, GObject)

FOUNDRY_AVAILABLE_IN_ALL
FoundryShortcutObserver *foundry_shortcut_observer_new    (GListModel              *model);
FOUNDRY_AVAILABLE_IN_ALL
const char              *foundry_shortcut_observer_lookup (FoundryShortcutObserver *self,
                                                           const char              *action_name);

G_END_DECLS
