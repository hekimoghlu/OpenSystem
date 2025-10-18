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

#include "config.h"

#include "gkm-attributes.h"
#include "gkm-log.h"
#include "gkm-test.h"
#include "gkm-util.h"

#include "pkcs11/pkcs11.h"
#include "pkcs11/pkcs11i.h"

#include <glib.h>

#include <string.h>

void
gkm_assertion_message_cmprv (const gchar *domain, const gchar *file, gint line,
                             const gchar *func, const gchar *expr,
                             CK_RV arg1, const gchar *cmp, CK_RV arg2)
{
	gchar *s;
	s = g_strdup_printf ("assertion failed (%s): (%s %s %s)", expr,
	                     gkm_log_rv (arg1), cmp, gkm_log_rv (arg2));
	g_assertion_message (domain, file, line, func, s);
	g_free (s);
}

void
gkm_assertion_message_cmpulong (const gchar *domain, const gchar *file, gint line,
                                const gchar *func, const gchar *expr,
                                CK_ULONG arg1, const gchar *cmp, CK_ULONG arg2)
{
	char *s = NULL;
	s = g_strdup_printf ("assertion failed (%s): (0x%08llx %s 0x%08llx)", expr,
	                     (long long unsigned)arg1, cmp, (long long unsigned)arg2);
	g_assertion_message (domain, file, line, func, s);
	g_free (s);
}
