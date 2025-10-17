/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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

#include "back-bdb.h"
#include "lber_pvt.h"
#include "lutil.h"


/* Congestion avoidance code
 * for Deadlock Rollback
 */

void
bdb_trans_backoff( int num_retries )
{
	int i;
	int delay = 0;
	int pow_retries = 1;
	unsigned long key = 0;
	unsigned long max_key = -1;
	struct timeval timeout;

	lutil_entropy( (unsigned char *) &key, sizeof( unsigned long ));

	for ( i = 0; i < num_retries; i++ ) {
		if ( i >= 5 ) break;
		pow_retries *= 4;
	}

	delay = 16384 * (key * (double) pow_retries / (double) max_key);
	delay = delay ? delay : 1;

	Debug( LDAP_DEBUG_TRACE,  "delay = %d, num_retries = %d\n", delay, num_retries, 0 );

	timeout.tv_sec = delay / 1000000;
	timeout.tv_usec = delay % 1000000;
	select( 0, NULL, NULL, NULL, &timeout );
}
