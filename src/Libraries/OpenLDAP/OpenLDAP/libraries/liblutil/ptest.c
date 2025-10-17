/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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

#include <ac/stdlib.h>

#include <ac/ctype.h>
#include <ac/signal.h>
#include <ac/socket.h>
#include <ac/string.h>
#include <ac/time.h>

#include <lber.h>

#include "lutil.h"

/*
 * Password Test Program
 */

static char *hash[] = {
#ifdef SLAP_AUTHPASSWD
	"SHA1", "MD5",
#else
#ifdef SLAPD_CRYPT
	"{CRYPT}",
#endif
	"{SSHA}", "{SMD5}",
	"{SHA}", "{MD5}",
	"{BOGUS}",
#endif
	NULL
};

static struct berval pw[] = {
	{ sizeof("secret")-1,			"secret" },
	{ sizeof("binary\0secret")-1,	"binary\0secret" },
	{ 0, NULL }
};

int
main( int argc, char *argv[] )
{
	int i, j, rc;
	struct berval *passwd;
#ifdef SLAP_AUTHPASSWD
	struct berval *salt;
#endif
	struct berval bad;
	bad.bv_val = "bad password";
	bad.bv_len = sizeof("bad password")-1;

	for( i= 0; hash[i]; i++ ) {
		for( j = 0; pw[j].bv_len; j++ ) {
#ifdef SLAP_AUTHPASSWD
			rc = lutil_authpasswd_hash( &pw[j],
				&passwd, &salt, hash[i] );

			if( rc )
#else
			passwd = lutil_passwd_hash( &pw[j], hash[i] );

			if( passwd == NULL )
#endif
			{
				printf("%s generate fail: %s (%d)\n", 
					hash[i], pw[j].bv_val, pw[j].bv_len );
				continue;
			}


#ifdef SLAP_AUTHPASSWD
			rc = lutil_authpasswd( &pw[j], passwd, salt, NULL );
#else
			rc = lutil_passwd( passwd, &pw[j], NULL );
#endif

			printf("%s (%d): %s (%d)\t(%d) %s\n",
				pw[j].bv_val, pw[j].bv_len, passwd->bv_val, passwd->bv_len,
				rc, rc == 0 ? "OKAY" : "BAD" );

#ifdef SLAP_AUTHPASSWD
			rc = lutil_authpasswd( passwd, salt, &bad, NULL );
#else
			rc = lutil_passwd( passwd, &bad, NULL );
#endif

			printf("%s (%d): %s (%d)\t(%d) %s\n",
				bad.bv_val, bad.bv_len, passwd->bv_val, passwd->bv_len,
				rc, rc != 0 ? "OKAY" : "BAD" );
		}

		printf("\n");
	}

	return EXIT_SUCCESS;
}
