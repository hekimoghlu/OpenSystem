/*
 * gnome-keyring
 *
 * Copyright (C) 2009 Stefan Walter
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef GKM_TIMER_H_
#define GKM_TIMER_H_

#include <glib.h>

#include "gkm-types.h"

typedef void    (*GkmTimerFunc)                (GkmTimer *timer,
                                                gpointer user_data);

GkmTimer*       gkm_timer_start                (GkmModule *module,
                                                glong seconds,
                                                GkmTimerFunc func,
                                                gpointer user_data);

void            gkm_timer_cancel               (GkmTimer *timer);

void            gkm_timer_initialize           (void);

void            gkm_timer_shutdown             (void);

#endif /* GKM_TIMER_H_ */
