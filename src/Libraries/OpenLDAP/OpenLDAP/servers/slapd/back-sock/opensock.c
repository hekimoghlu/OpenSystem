/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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

#include <ac/errno.h>
#include <ac/string.h>
#include <ac/socket.h>
#include <ac/unistd.h>

#include "slap.h"
#include "back-sock.h"

/*
 * FIXME: count the number of concurrent open sockets (since each thread
 * may open one). Perhaps block here if a soft limit is reached, and fail
 * if a hard limit reached
 */

FILE *
opensock(
    const char	*sockpath
)
{
	int	fd;
	FILE	*fp;
	struct sockaddr_un sockun;

	fd = socket(PF_UNIX, SOCK_STREAM, 0);
	if ( fd < 0 ) {
		Debug( LDAP_DEBUG_ANY, "socket create failed\n", 0, 0, 0 );
		return( NULL );
	}

	sockun.sun_family = AF_UNIX;
	sprintf(sockun.sun_path, "%.*s", (int)(sizeof(sockun.sun_path)-1),
		sockpath);
	if ( connect( fd, (struct sockaddr *)&sockun, sizeof(sockun) ) < 0 ) {
		Debug( LDAP_DEBUG_ANY, "socket connect(%s) failed\n",
			sockpath ? sockpath : "<null>", 0, 0 );
		close( fd );
		return( NULL );
	}

	if ( ( fp = fdopen( fd, "r+" ) ) == NULL ) {
		Debug( LDAP_DEBUG_ANY, "fdopen failed\n", 0, 0, 0 );
		close( fd );
		return( NULL );
	}

	return( fp );
}
