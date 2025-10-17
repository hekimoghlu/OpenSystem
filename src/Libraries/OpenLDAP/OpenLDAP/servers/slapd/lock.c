/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
/* Portions Copyright (c) 1995 Regents of the University of Michigan.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that this notice is preserved and that due credit is given
 * to the University of Michigan at Ann Arbor. The name of the University
 * may not be used to endorse or promote products derived from this
 * software without specific prior written permission. This software
 * is provided ``as is'' without express or implied warranty.
 */

#include "portable.h"

#include <stdio.h>

#include <ac/string.h>
#include <ac/socket.h>
#include <ac/time.h>
#include <ac/unistd.h>

#ifdef HAVE_SYS_FILE_H
#include <sys/file.h>
#endif

#include "slap.h"
#include <lutil.h>

FILE *
lock_fopen( const char *fname, const char *type, FILE **lfp )
{
	FILE	*fp;
	char	buf[MAXPATHLEN];

	/* open the lock file */
	snprintf( buf, sizeof buf, "%s.lock", fname );

	if ( (*lfp = fopen( buf, "w" )) == NULL ) {
		Debug( LDAP_DEBUG_ANY, "could not open \"%s\"\n", buf, 0, 0 );

		return( NULL );
	}

	/* acquire the lock */
	ldap_lockf( fileno(*lfp) );

	/* open the log file */
	if ( (fp = fopen( fname, type )) == NULL ) {
		Debug( LDAP_DEBUG_ANY, "could not open \"%s\"\n", fname, 0, 0 );

		ldap_unlockf( fileno(*lfp) );
		fclose( *lfp );
		*lfp = NULL;
		return( NULL );
	}

	return( fp );
}

int
lock_fclose( FILE *fp, FILE *lfp )
{
	int rc = fclose( fp );
	/* unlock */
	ldap_unlockf( fileno(lfp) );
	fclose( lfp );

	return( rc );
}
