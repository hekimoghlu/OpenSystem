/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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

#include "slap.h"
#include "back-bdb.h"

#if DB_VERSION_FULL < 0x04030000
void bdb_errcall( const char *pfx, char * msg )
#else
void bdb_errcall( const DB_ENV *env, const char *pfx, const char * msg )
#endif
{
#ifdef HAVE_EBCDIC
	if ( msg[0] > 0x7f )
		__etoa( msg );
#endif
	Debug( LDAP_DEBUG_ANY, "bdb(%s): %s\n", pfx, msg, 0 );
}

#if DB_VERSION_FULL >= 0x04030000
void bdb_msgcall( const DB_ENV *env, const char *msg )
{
#ifdef HAVE_EBCDIC
	if ( msg[0] > 0x7f )
		__etoa( msg );
#endif
	Debug( LDAP_DEBUG_TRACE, "bdb: %s\n", msg, 0, 0 );
}
#endif

#ifdef HAVE_EBCDIC

#undef db_strerror

/* Not re-entrant! */
char *ebcdic_dberror( int rc )
{
	static char msg[1024];

	strcpy( msg, db_strerror( rc ) );
	__etoa( msg );
	return msg;
}
#endif
