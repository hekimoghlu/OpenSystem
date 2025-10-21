/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 8; tab-width: 8 -*- */
/* test-module.c: A test PKCS#11 module implementation

   Copyright (C) 2009 Stefan Walter

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

#include "mock-module.h"

#include "egg/egg-secure-memory.h"

/* Include all the module entry points */
#include "gkm/gkm-module-ep.h"
GKM_DEFINE_MODULE (test_module, GKM_TYPE_MODULE);

#include "gkm/gkm-certificate.h"

EGG_SECURE_DEFINE_GLIB_GLOBALS ();

GkmModule*
mock_module_initialize_and_enter (void)
{
	CK_RV rv;

	gkm_crypto_initialize ();
	rv = test_module_function_list->C_Initialize (NULL);
	g_return_val_if_fail (rv == CKR_OK, NULL);

	g_return_val_if_fail (pkcs11_module, NULL);

	mock_module_enter ();
	return pkcs11_module;
}

void
mock_module_leave_and_finalize (void)
{
	CK_RV rv;

	mock_module_leave ();
	rv = test_module_function_list->C_Finalize (NULL);
	g_return_if_fail (rv == CKR_OK);
}

void
mock_module_leave (void)
{
	g_mutex_unlock (&pkcs11_module_mutex);
}

void
mock_module_enter (void)
{
	g_mutex_lock (&pkcs11_module_mutex);
}

GkmSession*
mock_module_open_session (gboolean writable)
{
	CK_ULONG flags = CKF_SERIAL_SESSION;
	CK_SESSION_HANDLE handle;
	GkmSession *session;
	CK_RV rv;

	if (writable)
		flags |= CKF_RW_SESSION;

	rv = gkm_module_C_OpenSession (pkcs11_module, 1, flags, NULL, NULL, &handle);
	g_assert (rv == CKR_OK);

	session = gkm_module_lookup_session (pkcs11_module, handle);
	g_assert (session);

	return session;
}

GkmObject*
mock_module_object_new (GkmSession *session)
{
	CK_BBOOL token = CK_FALSE;
	CK_OBJECT_CLASS klass = CKO_CERTIFICATE;
	CK_CERTIFICATE_TYPE type = CKC_X_509;
	GkmObject *object;

	gsize n_data;
	gchar *data;

	CK_ATTRIBUTE attrs[] = {
		{ CKA_VALUE, NULL, 0 },
		{ CKA_TOKEN, &token, sizeof (token) },
		{ CKA_CLASS, &klass, sizeof (klass) },
		{ CKA_CERTIFICATE_TYPE, &type, sizeof (type) },
	};

	if (!g_file_get_contents (SRCDIR "/pkcs11/gkm/fixtures/test-certificate-1.der", &data, &n_data, NULL))
		g_assert_not_reached ();

	attrs[0].pValue = data;
	attrs[0].ulValueLen = n_data;

	object = gkm_session_create_object_for_factory (session, GKM_FACTORY_CERTIFICATE, NULL,
	                                              attrs, G_N_ELEMENTS (attrs));
	if (object) /* Owned by storage */
		g_object_unref (object);

	g_free (data);
	return object;
}
