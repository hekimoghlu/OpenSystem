/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
/* Portions Copyright (c) 1990 Regents of the University of Michigan.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that this notice is preserved and that due credit is given
 * to the University of Michigan at Ann Arbor. The name of the University
 * may not be used to endorse or promote products derived from this
 * software without specific prior written permission. This software
 * is provided ``as is'' without express or implied warranty.
 */
/* ACKNOWLEDGEMENTS:
 * This work was originally developed by the University of Michigan
 * (as part of U-MICH LDAP).
 */

#include "portable.h"

#include <stdio.h>

#include <ac/stdlib.h>
#include <ac/string.h>
#include <ac/socket.h>
#include <ac/unistd.h>
#include <ac/errno.h>

#ifdef HAVE_CONSOLE_H
#include <console.h>
#endif

#include <lber.h>

static void usage( const char *name )
{
	fprintf( stderr, "usage: %s fmt\n", name );
}

int
main( int argc, char **argv )
{
	char *s;

	ber_tag_t	tag;
	ber_len_t	len;

	BerElement	*ber;
	Sockbuf		*sb;
	int		fd;

	/* enable debugging */
	int ival = -1;
	ber_set_option( NULL, LBER_OPT_DEBUG_LEVEL, &ival );

	if ( argc < 2 ) {
		usage( argv[0] );
		return( EXIT_FAILURE );
	}

#ifdef HAVE_CONSOLE_H
	ccommand( &argv );
	cshow( stdout );
#endif

	sb = ber_sockbuf_alloc();
	fd = fileno( stdin );
	ber_sockbuf_add_io( sb, &ber_sockbuf_io_fd, LBER_SBIOD_LEVEL_PROVIDER,
		(void *)&fd );

	ber = ber_alloc_t(LBER_USE_DER);
	if( ber == NULL ) {
		perror( "ber_alloc_t" );
		return( EXIT_FAILURE );
	}

	for (;;) {
		tag = ber_get_next( sb, &len, ber);
		if( tag != LBER_ERROR ) break;

		if( errno == EWOULDBLOCK ) continue;
		if( errno == EAGAIN ) continue;

		perror( "ber_get_next" );
		return( EXIT_FAILURE );
	}

	printf("decode: message tag 0x%lx and length %ld\n",
		(unsigned long) tag, (long) len );

	for( s = argv[1]; *s; s++ ) {
		char buf[128];
		char fmt[2];
		fmt[0] = *s;
		fmt[1] = '\0';

		printf("decode: format %s\n", fmt );
		len = sizeof(buf);
		tag = ber_scanf( ber, fmt, &buf[0], &len );

		if( tag == LBER_ERROR ) {
			perror( "ber_scanf" );
			return( EXIT_FAILURE );
		}
	}

	ber_sockbuf_free( sb );
	return( EXIT_SUCCESS );
}
