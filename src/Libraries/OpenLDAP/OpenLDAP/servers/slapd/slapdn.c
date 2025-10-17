/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
 * Copyright 2004-2011 The OpenLDAP Foundation.
 * Portions Copyright 2004 Pierangelo Masarati.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */
/* ACKNOWLEDGEMENTS:
 * This work was initially developed by Pierangelo Masarati for inclusion
 * in OpenLDAP Software.
 */

#include "portable.h"

#include <stdio.h>

#include <ac/stdlib.h>

#include <ac/ctype.h>
#include <ac/string.h>
#include <ac/socket.h>
#include <ac/unistd.h>

#include <lber.h>
#include <ldif.h>
#include <lutil.h>

#include "slapcommon.h"

int
slapdn( int argc, char **argv )
{
	int			rc = 0;
	const char		*progname = "slapdn";

	slap_tool_init( progname, SLAPDN, argc, argv );

	argv = &argv[ optind ];
	argc -= optind;

	for ( ; argc--; argv++ ) {
		struct berval	dn,
				pdn = BER_BVNULL,
				ndn = BER_BVNULL;

		ber_str2bv( argv[ 0 ], 0, 0, &dn );

		switch ( dn_mode ) {
		case SLAP_TOOL_LDAPDN_PRETTY:
			rc = dnPretty( NULL, &dn, &pdn, NULL );
			break;

		case SLAP_TOOL_LDAPDN_NORMAL:
			rc = dnNormalize( 0, NULL, NULL, &dn, &ndn, NULL );
			break;

		default:
			rc = dnPrettyNormal( NULL, &dn, &pdn, &ndn, NULL );
			break;
		}

		if ( rc != LDAP_SUCCESS ) {
			fprintf( stderr, "DN: <%s> check failed %d (%s)\n",
					dn.bv_val, rc,
					ldap_err2string( rc ) );
			if ( !continuemode ) {
				rc = -1;
				break;
			}
			
		} else {
			switch ( dn_mode ) {
			case SLAP_TOOL_LDAPDN_PRETTY:
				printf( "%s\n", pdn.bv_val );
				break;

			case SLAP_TOOL_LDAPDN_NORMAL:
				printf( "%s\n", ndn.bv_val );
				break;

			default:
				printf( "DN: <%s> check succeeded\n"
						"normalized: <%s>\n"
						"pretty:     <%s>\n",
						dn.bv_val,
						ndn.bv_val, pdn.bv_val );
				break;
			}

			ch_free( ndn.bv_val );
			ch_free( pdn.bv_val );
		}
	}
	
	if ( slap_tool_destroy())
		rc = EXIT_FAILURE;

	return rc;
}
