/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
 * Copyright 2007-2011 The OpenLDAP Foundation.
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
 * This work was initially developed by Brian Candler for inclusion
 * in OpenLDAP Software.
 */

#include "portable.h"

#include <stdio.h>

#include <ac/socket.h>
#include <ac/string.h>

#include "slap.h"
#include "back-sock.h"

int
sock_back_unbind(
    Operation		*op,
    SlapReply		*rs
)
{
	struct sockinfo	*si = (struct sockinfo *) op->o_bd->be_private;
	FILE			*fp;

	if ( (fp = opensock( si->si_sockpath )) == NULL ) {
		send_ldap_error( op, rs, LDAP_OTHER,
		    "could not open socket" );
		return( -1 );
	}

	/* write out the request to the unbind process */
	fprintf( fp, "UNBIND\n" );
	fprintf( fp, "msgid: %ld\n", (long) op->o_msgid );
	sock_print_conn( fp, op->o_conn, si );
	sock_print_suffixes( fp, op->o_bd );
	fprintf( fp, "\n" );

	/* no response to unbind */
	fclose( fp );

	return 0;
}
