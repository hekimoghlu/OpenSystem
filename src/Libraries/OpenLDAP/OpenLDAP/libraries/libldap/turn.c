/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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
 * Copyright 2005-2011 The OpenLDAP Foundation.
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
/* ACKNOWLEDGEMENTS:
 * This program was orignally developed by Kurt D. Zeilenga for inclusion in
 * OpenLDAP Software.
 */

/*
 * LDAPv3 Turn Operation Request
 */

#include "portable.h"

#include <stdio.h>
#include <ac/stdlib.h>

#include <ac/socket.h>
#include <ac/string.h>
#include <ac/time.h>

#include "ldap-int.h"
#include "ldap_log.h"

int
ldap_turn(
	LDAP *ld,
	int mutual,
	LDAP_CONST char* identifier,
	LDAPControl **sctrls,
	LDAPControl **cctrls,
	int *msgidp )
{
#ifdef LDAP_EXOP_X_TURN
	BerElement *turnvalber = NULL;
	struct berval *turnvalp = NULL;
	int rc;

	turnvalber = ber_alloc_t( LBER_USE_DER );
	if( mutual ) {
		ber_printf( turnvalber, "{bs}", mutual, identifier );
	} else {
		ber_printf( turnvalber, "{s}", identifier );
	}
	ber_flatten( turnvalber, &turnvalp );

	rc = ldap_extended_operation( ld, LDAP_EXOP_X_TURN,
			turnvalp, sctrls, cctrls, msgidp );
	ber_free( turnvalber, 1 );
	return rc;
#else
	return LDAP_CONTROL_NOT_FOUND;
#endif
}

int
ldap_turn_s(
	LDAP *ld,
	int mutual,
	LDAP_CONST char* identifier,
	LDAPControl **sctrls,
	LDAPControl **cctrls )
{
#ifdef LDAP_EXOP_X_TURN
	BerElement *turnvalber = NULL;
	struct berval *turnvalp = NULL;
	int rc;

	turnvalber = ber_alloc_t( LBER_USE_DER );
	if( mutual ) {
		ber_printf( turnvalber, "{bs}", 0xFF, identifier );
	} else {
		ber_printf( turnvalber, "{s}", identifier );
	}
	ber_flatten( turnvalber, &turnvalp );

	rc = ldap_extended_operation_s( ld, LDAP_EXOP_X_TURN,
			turnvalp, sctrls, cctrls, NULL, NULL );
	ber_free( turnvalber, 1 );
	return rc;
#else
	return LDAP_CONTROL_NOT_FOUND;
#endif
}

