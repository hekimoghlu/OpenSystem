/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#include <ac/string.h>
#include <ac/time.h>

#include "ldap-int.h"

int
ldap_create_assertion_control_value(
	LDAP		*ld,
	char		*assertion,
	struct berval	*value )
{
	BerElement		*ber = NULL;
	int			err;

	if ( assertion == NULL || assertion[ 0 ] == '\0' ) {
		ld->ld_errno = LDAP_PARAM_ERROR;
		return ld->ld_errno;
	}

	if ( value == NULL ) {
		ld->ld_errno = LDAP_PARAM_ERROR;
		return ld->ld_errno;
	}

	BER_BVZERO( value );

	ber = ldap_alloc_ber_with_options( ld );
	if ( ber == NULL ) {
		ld->ld_errno = LDAP_NO_MEMORY;
		return ld->ld_errno;
	}

	err = ldap_pvt_put_filter( ber, assertion );
	if ( err < 0 ) {
		ld->ld_errno = LDAP_ENCODING_ERROR;
		goto done;
	}

	err = ber_flatten2( ber, value, 1 );
	if ( err < 0 ) {
		ld->ld_errno = LDAP_NO_MEMORY;
		goto done;
	}

done:;
	if ( ber != NULL ) {
		ber_free( ber, 1 );
	}

	return ld->ld_errno;
}

int
ldap_create_assertion_control(
	LDAP		*ld,
	char		*assertion,
	int		iscritical,
	LDAPControl	**ctrlp )
{
	struct berval	value;

	if ( ctrlp == NULL ) {
		ld->ld_errno = LDAP_PARAM_ERROR;
		return ld->ld_errno;
	}

	ld->ld_errno = ldap_create_assertion_control_value( ld,
		assertion, &value );
	if ( ld->ld_errno == LDAP_SUCCESS ) {
		ld->ld_errno = ldap_control_create( LDAP_CONTROL_ASSERT,
			iscritical, &value, 0, ctrlp );
		if ( ld->ld_errno != LDAP_SUCCESS ) {
			LDAP_FREE( value.bv_val );
		}
	}

	return ld->ld_errno;
}

