/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#include <ac/ctype.h>
#include <ac/string.h>

#ifdef HAVE_FSTAT
#include <sys/types.h>
#include <sys/stat.h>
#endif /* HAVE_FSTAT */

#include <lber.h>
#include <lutil.h>

/* Get a password from a file. */
int
lutil_get_filed_password(
	const char *filename,
	struct berval *passwd )
{
	size_t nread, nleft, nr;
	FILE *f = fopen( filename, "r" );

	if( f == NULL ) {
		perror( filename );
		return -1;
	}

	passwd->bv_val = NULL;
	passwd->bv_len = 4096;

#ifdef HAVE_FSTAT
	{
		struct stat sb;
		if ( fstat( fileno( f ), &sb ) == 0 ) {
			if( sb.st_mode & 006 ) {
				fprintf( stderr, _("Warning: Password file %s"
					" is publicly readable/writeable\n"),
					filename );
			}

			if ( sb.st_size )
				passwd->bv_len = sb.st_size;
		}
	}
#endif /* HAVE_FSTAT */

	passwd->bv_val = (char *) ber_memalloc( passwd->bv_len + 1 );
	if( passwd->bv_val == NULL ) {
		perror( filename );
		fclose( f );
		return -1;
	}

	nread = 0;
	nleft = passwd->bv_len;
	do {
		if( nleft == 0 ) {
			/* double the buffer size */
			char *p = (char *) ber_memrealloc( passwd->bv_val,
				2 * passwd->bv_len + 1 );
			if( p == NULL ) {
				ber_memfree( passwd->bv_val );
				passwd->bv_val = NULL;
				passwd->bv_len = 0;
				fclose( f );
				return -1;
			}
			nleft = passwd->bv_len;
			passwd->bv_len *= 2;
			passwd->bv_val = p;
		}

		nr = fread( &passwd->bv_val[nread], 1, nleft, f );

		if( nr < nleft && ferror( f ) ) {
			ber_memfree( passwd->bv_val );
			passwd->bv_val = NULL;
			passwd->bv_len = 0;
			fclose( f );
			return -1;
		}

		nread += nr;
		nleft -= nr;
	} while ( !feof(f) );

	passwd->bv_len = nread;
	passwd->bv_val[nread] = '\0';

	fclose( f );
	return 0;
}
