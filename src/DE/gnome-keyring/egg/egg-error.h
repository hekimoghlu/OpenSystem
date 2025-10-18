/*
 * gnome-keyring
 *
 * Copyright (C) 2010 Stefan Walter
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

#ifndef EGG_ERROR_H_
#define EGG_ERROR_H_

#include <glib.h>

static inline const gchar*
egg_error_message (GError *error)
{
	g_return_val_if_fail (error, "(unknown)");
	return error->message ? error->message : "(null)";
}

#endif /* EGG_ERROR_H_ */
