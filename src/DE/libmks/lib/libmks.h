/*
 * libmks.h
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
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

#include <glib.h>

G_BEGIN_DECLS

#define MKS_INSIDE
# include "mks-device.h"
# include "mks-display.h"
# include "mks-enums.h"
# include "mks-init.h"
# include "mks-keyboard.h"
# include "mks-mouse.h"
# include "mks-screen.h"
# include "mks-screen-attributes.h"
# include "mks-session.h"
# include "mks-touchable.h"
# include "mks-types.h"
# include "mks-version.h"
# include "mks-version-macros.h"
#undef MKS_INSIDE

G_END_DECLS
