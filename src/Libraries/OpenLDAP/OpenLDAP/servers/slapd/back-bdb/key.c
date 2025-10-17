/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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

#include <ac/string.h>
#include <ac/socket.h>

#include "slap.h"
#include "back-bdb.h"
#include "idl.h"

/* read a key */
int
bdb_key_read(
	Backend	*be,
	DB *db,
	DB_TXN *txn,
	struct berval *k,
	ID *ids,
	DBC **saved_cursor,
	int get_flag
)
{
	int rc;
	DBT key;

	Debug( LDAP_DEBUG_TRACE, "=> key_read\n", 0, 0, 0 );

	DBTzero( &key );
	bv2DBT(k,&key);
	key.ulen = key.size;
	key.flags = DB_DBT_USERMEM;

	rc = bdb_idl_fetch_key( be, db, txn, &key, ids, saved_cursor, get_flag );

	if( rc != LDAP_SUCCESS ) {
		Debug( LDAP_DEBUG_TRACE, "<= bdb_index_read: failed (%d)\n",
			rc, 0, 0 );
	} else {
		Debug( LDAP_DEBUG_TRACE, "<= bdb_index_read %ld candidates\n",
			(long) BDB_IDL_N(ids), 0, 0 );
	}

	return rc;
}

/* Add or remove stuff from index files */
int
bdb_key_change(
	Backend *be,
	DB *db,
	DB_TXN *txn,
	struct berval *k,
	ID id,
	int op
)
{
	int	rc;
	DBT	key;

	Debug( LDAP_DEBUG_TRACE, "=> key_change(%s,%lx)\n",
		op == SLAP_INDEX_ADD_OP ? "ADD":"DELETE", (long) id, 0 );

	DBTzero( &key );
	bv2DBT(k,&key);
	key.ulen = key.size;
	key.flags = DB_DBT_USERMEM;

	if (op == SLAP_INDEX_ADD_OP) {
		/* Add values */

#ifdef BDB_TOOL_IDL_CACHING
		if ( slapMode & SLAP_TOOL_QUICK )
			rc = bdb_tool_idl_add( be, db, txn, &key, id );
		else
#endif
			rc = bdb_idl_insert_key( be, db, txn, &key, id );
		if ( rc == DB_KEYEXIST ) rc = 0;
	} else {
		/* Delete values */
		rc = bdb_idl_delete_key( be, db, txn, &key, id );
		if ( rc == DB_NOTFOUND ) rc = 0;
	}

	Debug( LDAP_DEBUG_TRACE, "<= key_change %d\n", rc, 0, 0 );

	return rc;
}
