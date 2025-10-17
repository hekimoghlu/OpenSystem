/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
 * Copyright 2000-2011 The OpenLDAP Foundation.
 * Portions Copyright 2000-2003 Kurt D. Zeilenga.
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
 * This work was originally developed by Kurt D. Zeilenga for inclusion
 * in OpenLDAP Software.
 */


#include "portable.h"

#include <stdio.h>

#include <ac/socket.h>
#include <ac/string.h>

#include "slap.h"
#include "proto-dnssrv.h"

int
dnssrv_back_bind(
	Operation	*op,
	SlapReply	*rs )
{
	Debug( LDAP_DEBUG_TRACE, "DNSSRV: bind dn=\"%s\" (%d)\n",
		BER_BVISNULL( &op->o_req_dn ) ? "" : op->o_req_dn.bv_val, 
		op->orb_method, 0 );

	/* allow rootdn as a means to auth without the need to actually
 	 * contact the proxied DSA */
	switch ( be_rootdn_bind( op, NULL ) ) {
	case LDAP_SUCCESS:
		/* frontend will send result */
		return rs->sr_err;

	default:
		/* treat failure and like any other bind, otherwise
		 * it could reveal the DN of the rootdn */
		break;
	}

	if ( !BER_BVISNULL( &op->orb_cred ) &&
		!BER_BVISEMPTY( &op->orb_cred ) )
	{
		/* simple bind */
		Statslog( LDAP_DEBUG_STATS,
		   	"%s DNSSRV BIND dn=\"%s\" provided cleartext passwd\n",
	   		op->o_log_prefix,
			BER_BVISNULL( &op->o_req_dn ) ? "" : op->o_req_dn.bv_val , 0, 0, 0 );

		send_ldap_error( op, rs, LDAP_UNWILLING_TO_PERFORM,
			"you shouldn't send strangers your password" );

	} else {
		/* unauthenticated bind */
		/* NOTE: we're not going to get here anyway:
		 * unauthenticated bind is dealt with by the frontend */
		Debug( LDAP_DEBUG_TRACE, "DNSSRV: BIND dn=\"%s\"\n",
			BER_BVISNULL( &op->o_req_dn ) ? "" : op->o_req_dn.bv_val, 0, 0 );

		send_ldap_error( op, rs, LDAP_UNWILLING_TO_PERFORM,
			"anonymous bind expected" );
	}

	return 1;
}
