/* -*- Mode: C; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 8 -*-
 *
 * Copyright 2008 Red Hat, Inc.
 * Copyright 2007 William Jon McCann <mccann@jhu.edu>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * Written by: Ray Strode
 *             William Jon McCann
 */

#ifndef __GDM_GREETER_SESSIONS_H
#define __GDM_GREETER_SESSIONS_H

#include <glib.h>

G_BEGIN_DECLS

char **                gdm_get_session_ids (void);
char *                 gdm_get_session_name_and_description (const char  *id,
                                                             char       **description);

G_END_DECLS

#endif /* __GDM_SESSION_H */
