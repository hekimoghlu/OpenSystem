/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 8; tab-width: 8 -*- */
/* gkm-gnome2-standalone.h - The user-store PKCS#11 code as a standalone module

   Copyright (C) 2008, Stef Walter

   The Gnome Keyring Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The Gnome Keyring Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the Gnome Library; see the file COPYING.LIB.  If not,
   <http://www.gnu.org/licenses/>.

   Author: Stef Walter <stef@memberwebs.com>
*/

#include "config.h"

#include "gkm-xdg-store.h"

#include "gkm/gkm-crypto.h"

#include "egg/egg-secure-memory.h"

#include "pkcs11/pkcs11.h"

#include <glib-object.h>
#include <gmodule.h>

/* Module callbacks for secure memory */
EGG_SECURE_DEFINE_GLIB_GLOBALS ();

const gchar* g_module_check_init (GModule *module);

const gchar*
g_module_check_init (GModule *module)
{
	/*
	 * We can't handle being unloaded due to the gobject
	 * types we register
	 */

	g_module_make_resident (module);
	return NULL;
}

CK_RV
C_GetFunctionList (CK_FUNCTION_LIST_PTR_PTR list)
{
	if (!list)
		return CKR_ARGUMENTS_BAD;

#if !GLIB_CHECK_VERSION(2,35,0)
	g_type_init ();
#endif

	*list = gkm_xdg_store_get_functions ();
	return CKR_OK;
}
