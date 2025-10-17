/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
/* Portions Copyright (c) 1993 Regents of the University of Michigan.
 * All rights reserved.
 */

/*
 *	BindRequest ::= SEQUENCE {
 *		version		INTEGER,
 *		name		DistinguishedName,	 -- who
 *		authentication	CHOICE {
 *			simple		[0] OCTET STRING -- passwd
 *			krbv42ldap	[1] OCTET STRING  -- OBSOLETE
 *			krbv42dsa	[2] OCTET STRING  -- OBSOLETE
 *			sasl		[3] SaslCredentials	-- LDAPv3
 *		}
 *	}
 *
 *	BindResponse ::= SEQUENCE {
 *		COMPONENTS OF LDAPResult,
 *		serverSaslCreds		OCTET STRING OPTIONAL -- LDAPv3
 *	}
 *
 */

#include "portable.h"

#include <stdio.h>

#include <ac/socket.h>
#include <ac/string.h>
#include <ac/time.h>

#include "ldap-int.h"

/*
 * ldap_simple_bind - bind to the ldap server (and X.500).  The dn and
 * password of the entry to which to bind are supplied.  The message id
 * of the request initiated is returned.
 *
 * Example:
 *	ldap_simple_bind( ld, "cn=manager, o=university of michigan, c=us",
 *	    "secret" )
 */

int
ldap_simple_bind(
	LDAP *ld,
	LDAP_CONST char *dn,
	LDAP_CONST char *passwd )
{
	int rc;
	int msgid;
	struct berval cred;

	Debug( LDAP_DEBUG_TRACE, "ldap_simple_bind\n", 0, 0, 0 );

	assert( ld != NULL );
	assert( LDAP_VALID( ld ) );

	if ( passwd != NULL ) {
		cred.bv_val = (char *) passwd;
		cred.bv_len = strlen( passwd );
	} else {
		cred.bv_val = "";
		cred.bv_len = 0;
	}

	rc = ldap_sasl_bind( ld, dn, LDAP_SASL_SIMPLE, &cred,
		NULL, NULL, &msgid );

	return rc == LDAP_SUCCESS ? msgid : -1;
}

/*
 * ldap_simple_bind - bind to the ldap server (and X.500) using simple
 * authentication.  The dn and password of the entry to which to bind are
 * supplied.  LDAP_SUCCESS is returned upon success, the ldap error code
 * otherwise.
 *
 * Example:
 *	ldap_simple_bind_s( ld, "cn=manager, o=university of michigan, c=us",
 *	    "secret" )
 */

int
ldap_simple_bind_s( LDAP *ld, LDAP_CONST char *dn, LDAP_CONST char *passwd )
{
	struct berval cred;

	Debug( LDAP_DEBUG_TRACE, "ldap_simple_bind_s\n", 0, 0, 0 );

	if ( passwd != NULL ) {
		cred.bv_val = (char *) passwd;
		cred.bv_len = strlen( passwd );
	} else {
		cred.bv_val = "";
		cred.bv_len = 0;
	}

	return ldap_sasl_bind_s( ld, dn, LDAP_SASL_SIMPLE, &cred,
		NULL, NULL, NULL );
}
