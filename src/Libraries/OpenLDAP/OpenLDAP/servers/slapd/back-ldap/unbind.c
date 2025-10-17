/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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
 * Copyright 1999-2011 The OpenLDAP Foundation.
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

#include <ac/errno.h>
#include <ac/socket.h>
#include <ac/string.h>

#include "slap.h"
#include "back-ldap.h"

int
ldap_back_conn_destroy(
		Backend		*be,
		Connection	*conn
)
{
	ldapinfo_t	*li = (ldapinfo_t *) be->be_private;
	ldapconn_t	*lc = NULL, lc_curr;

	Debug( LDAP_DEBUG_TRACE,
		"=>ldap_back_conn_destroy: fetching conn %ld\n",
		conn->c_connid, 0, 0 );

	lc_curr.lc_conn = conn;
	
	ldap_pvt_thread_mutex_lock( &li->li_conninfo.lai_mutex );
#if LDAP_BACK_PRINT_CONNTREE > 0
	ldap_back_print_conntree( li, ">>> ldap_back_conn_destroy" );
#endif /* LDAP_BACK_PRINT_CONNTREE */
	while ( ( lc = avl_delete( &li->li_conninfo.lai_tree, (caddr_t)&lc_curr, ldap_back_conn_cmp ) ) != NULL )
	{
		assert( !LDAP_BACK_PCONN_ISPRIV( lc ) );
		Debug( LDAP_DEBUG_TRACE,
			"=>ldap_back_conn_destroy: destroying conn %lu "
			"refcnt=%d flags=0x%08x\n",
			lc->lc_conn->c_connid, lc->lc_refcnt, lc->lc_lcflags );

		if ( lc->lc_refcnt > 0 ) {
			/* someone else might be accessing the connection;
			 * mark for deletion */
			LDAP_BACK_CONN_CACHED_CLEAR( lc );
			LDAP_BACK_CONN_TAINTED_SET( lc );

		} else {
			ldap_back_conn_free( lc );
		}
	}
#if LDAP_BACK_PRINT_CONNTREE > 0
	ldap_back_print_conntree( li, "<<< ldap_back_conn_destroy" );
#endif /* LDAP_BACK_PRINT_CONNTREE */
	ldap_pvt_thread_mutex_unlock( &li->li_conninfo.lai_mutex );

	return 0;
}
