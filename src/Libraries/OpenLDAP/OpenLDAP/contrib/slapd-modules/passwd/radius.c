/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 1998-2011 The OpenLDAP Foundation.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in the file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */

#include "portable.h"

#include <stdio.h>

#include <lber.h>
#include <lber_pvt.h>	/* BER_BVC definition */
#include "lutil.h"
#include <ldap_pvt_thread.h>
#include <ac/string.h>
#include <ac/unistd.h>

#include <radlib.h>

static LUTIL_PASSWD_CHK_FUNC chk_radius;
static const struct berval scheme = BER_BVC("{RADIUS}");
static char *config_filename;
static ldap_pvt_thread_mutex_t libradius_mutex;

static int
chk_radius(
	const struct berval	*sc,
	const struct berval	*passwd,
	const struct berval	*cred,
	const char		**text )
{
	unsigned int		i;
	int			rc = LUTIL_PASSWD_ERR;

	struct rad_handle	*h = NULL;

	for ( i = 0; i < cred->bv_len; i++ ) {
		if ( cred->bv_val[ i ] == '\0' ) {
			return LUTIL_PASSWD_ERR;	/* NUL character in cred */
		}
	}

	if ( cred->bv_val[ i ] != '\0' ) {
		return LUTIL_PASSWD_ERR;	/* cred must behave like a string */
	}

	for ( i = 0; i < passwd->bv_len; i++ ) {
		if ( passwd->bv_val[ i ] == '\0' ) {
			return LUTIL_PASSWD_ERR;	/* NUL character in password */
		}
	}

	if ( passwd->bv_val[ i ] != '\0' ) {
		return LUTIL_PASSWD_ERR;	/* passwd must behave like a string */
	}

	ldap_pvt_thread_mutex_lock( &libradius_mutex );

	h = rad_auth_open();
	if ( h == NULL ) {
		ldap_pvt_thread_mutex_unlock( &libradius_mutex );
		return LUTIL_PASSWD_ERR;
	}

	if ( rad_config( h, config_filename ) != 0 ) {
		goto done;
	}

	if ( rad_create_request( h, RAD_ACCESS_REQUEST ) ) {
		goto done;
	}

	if ( rad_put_string( h, RAD_USER_NAME, passwd->bv_val ) != 0 ) {
		goto done;
	}

	if ( rad_put_string( h, RAD_USER_PASSWORD, cred->bv_val ) != 0 ) {
		goto done;
	}

	switch ( rad_send_request( h ) ) {
	case RAD_ACCESS_ACCEPT:
		rc = LUTIL_PASSWD_OK;
		break;

	case RAD_ACCESS_REJECT:
		rc = LUTIL_PASSWD_ERR;
		break;

	case RAD_ACCESS_CHALLENGE:
		rc = LUTIL_PASSWD_ERR;
		break;

	case -1:
		/* no valid response is received */
		break;
	}

done:;
	rad_close( h );

	ldap_pvt_thread_mutex_unlock( &libradius_mutex );
	return rc;
}

int
term_module()
{
	return ldap_pvt_thread_mutex_destroy( &libradius_mutex );
}

int
init_module( int argc, char *argv[] )
{
	int	i;

	for ( i = 0; i < argc; i++ ) {
		if ( strncasecmp( argv[ i ], "config=", STRLENOF( "config=" ) ) == 0 ) {
			/* FIXME: what if multiple loads of same module?
			 * does it make sense (e.g. override an existing one)? */
			if ( config_filename == NULL ) {
				config_filename = ber_strdup( &argv[ i ][ STRLENOF( "config=" ) ] );
			}

		} else {
			fprintf( stderr, "init_module(radius): unknown arg#%d=\"%s\".\n",
				i, argv[ i ] );
			return 1;
		}
	}

	ldap_pvt_thread_mutex_init( &libradius_mutex );

	return lutil_passwd_add( (struct berval *)&scheme, chk_radius, NULL );
}
