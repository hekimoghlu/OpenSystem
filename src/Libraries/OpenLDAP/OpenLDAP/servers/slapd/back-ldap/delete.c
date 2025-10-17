/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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
/* $OpenLDAP$ */
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 2003-2011 The OpenLDAP Foundation.
 * Portions Copyright 1999-2003 Howard Chu.
 * Portions Copyright 2000-2003 Pierangelo Masarati.
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
 * This work was initially developed by the Howard Chu for inclusion
 * in OpenLDAP Software and subsequently enhanced by Pierangelo
 * Masarati.
 */

#include "portable.h"

#include <stdio.h>

#include <ac/string.h>
#include <ac/socket.h>

#include "slap.h"
#include "back-ldap.h"

int
ldap_back_delete(
		Operation	*op,
		SlapReply	*rs )
{
	ldapinfo_t		*li = (ldapinfo_t *)op->o_bd->be_private;

	ldapconn_t		*lc = NULL;
	ber_int_t		msgid;
	LDAPControl		**ctrls = NULL;
	ldap_back_send_t	retrying = LDAP_BACK_RETRYING;
	int			rc = LDAP_SUCCESS;

	if ( !ldap_back_dobind( &lc, op, rs, LDAP_BACK_SENDERR ) ) {
		return rs->sr_err;
	}

retry:
	ctrls = op->o_ctrls;
	rc = ldap_back_controls_add( op, rs, lc, &ctrls );
	if ( rc != LDAP_SUCCESS ) {
		send_ldap_result( op, rs );
		rc = rs->sr_err;
		goto cleanup;
	}

	rs->sr_err = ldap_delete_ext( lc->lc_ld, op->o_req_dn.bv_val,
			ctrls, NULL, &msgid );
	rc = ldap_back_op_result( lc, op, rs, msgid,
		li->li_timeout[ SLAP_OP_DELETE ],
		( LDAP_BACK_SENDRESULT | retrying ) );
	if ( rs->sr_err == LDAP_UNAVAILABLE && retrying ) {
		retrying &= ~LDAP_BACK_RETRYING;
		if ( ldap_back_retry( &lc, op, rs, LDAP_BACK_SENDERR ) ) {
			/* if the identity changed, there might be need to re-authz */
			(void)ldap_back_controls_free( op, rs, &ctrls );
			goto retry;
		}
	}

cleanup:
	(void)ldap_back_controls_free( op, rs, &ctrls );

	if ( lc != NULL ) {
		ldap_back_release_conn( li, lc );
	}

	return rc;
}
