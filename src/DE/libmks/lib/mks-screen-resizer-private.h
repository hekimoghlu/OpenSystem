/*
 * mks-screen-resizer-private.h
 *
 * Copyright 2023 Bilal Elmoussaoui <belmouss@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "mks-screen.h"

G_BEGIN_DECLS

#define MKS_TYPE_SCREEN_RESIZER (mks_screen_resizer_get_type())

G_DECLARE_FINAL_TYPE (MksScreenResizer, mks_screen_resizer, MKS, SCREEN_RESIZER, GObject)

MksScreenResizer *mks_screen_resizer_new          (void);
void              mks_screen_resizer_set_screen   (MksScreenResizer *self,
                                                   MksScreen        *screen);
void              mks_screen_resizer_queue_resize (MksScreenResizer    *self,
                                                   MksScreenAttributes *attributes);

G_END_DECLS
