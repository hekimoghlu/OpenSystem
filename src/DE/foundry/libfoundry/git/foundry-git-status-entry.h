/* foundry-git-status-entry.h
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <glib-object.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_GIT_STATUS_ENTRY (foundry_git_status_entry_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryGitStatusEntry, foundry_git_status_entry, FOUNDRY, GIT_STATUS_ENTRY, GObject)

FOUNDRY_AVAILABLE_IN_ALL
char     *foundry_git_status_entry_dup_path             (FoundryGitStatusEntry *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean  foundry_git_status_entry_has_staged_changed   (FoundryGitStatusEntry *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean  foundry_git_status_entry_has_unstaged_changed (FoundryGitStatusEntry *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean  foundry_git_status_entry_is_new_file          (FoundryGitStatusEntry *self);
FOUNDRY_AVAILABLE_IN_ALL
GIcon    *foundry_git_status_entry_dup_icon             (FoundryGitStatusEntry *self);

G_END_DECLS
