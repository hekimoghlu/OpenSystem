/* foundry-shortcut-info.h
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

#include <foundry.h>

G_BEGIN_DECLS

typedef struct _FoundryShortcutInfo FoundryShortcutInfo;

typedef void (*FoundryShortcutInfoFunc) (const FoundryShortcutInfo *info,
                                         gpointer                   user_data);

FOUNDRY_AVAILABLE_IN_ALL
void        foundry_shortcut_info_foreach           (GListModel                *shortcuts,
                                                     FoundryShortcutInfoFunc    func,
                                                     gpointer                   func_data);
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_shortcut_info_get_id            (const FoundryShortcutInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_shortcut_info_get_icon_name     (const FoundryShortcutInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_shortcut_info_get_accelerator   (const FoundryShortcutInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_shortcut_info_get_action_name   (const FoundryShortcutInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
GVariant   *foundry_shortcut_info_get_action_target (const FoundryShortcutInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_shortcut_info_get_page          (const FoundryShortcutInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_shortcut_info_get_group         (const FoundryShortcutInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_shortcut_info_get_title         (const FoundryShortcutInfo *self);
FOUNDRY_AVAILABLE_IN_ALL
const char *foundry_shortcut_info_get_subtitle      (const FoundryShortcutInfo *self);

G_END_DECLS
