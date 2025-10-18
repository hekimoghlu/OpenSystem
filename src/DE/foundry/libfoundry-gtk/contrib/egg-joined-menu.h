/* egg-joined-menu.h
 *
 * Copyright 2017-2025 Christian Hergert <chergert@redhat.com>
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

G_BEGIN_DECLS

#define EGG_TYPE_JOINED_MENU (egg_joined_menu_get_type())

G_DECLARE_FINAL_TYPE (EggJoinedMenu, egg_joined_menu, EGG, JOINED_MENU, GMenuModel)

EggJoinedMenu *egg_joined_menu_new          (void);
guint          egg_joined_menu_get_n_joined (EggJoinedMenu *self);
void           egg_joined_menu_append_menu  (EggJoinedMenu *self,
                                             GMenuModel    *model);
void           egg_joined_menu_prepend_menu (EggJoinedMenu *self,
                                             GMenuModel    *model);
void           egg_joined_menu_remove_menu  (EggJoinedMenu *self,
                                             GMenuModel    *model);
void           egg_joined_menu_remove_index (EggJoinedMenu *self,
                                             guint          index);

G_END_DECLS
