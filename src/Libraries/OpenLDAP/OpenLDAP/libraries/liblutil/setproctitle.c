/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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
/* Portions Copyright (c) 1990,1991 Regents of the University of Michigan.
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

#ifndef HAVE_SETPROCTITLE

#include <stdio.h>

#include <ac/stdlib.h>

#include <ac/setproctitle.h>
#include <ac/string.h>
#include <ac/stdarg.h>

char	**Argv;		/* pointer to original (main's) argv */
int	Argc;		/* original argc */

/*
 * takes a printf-style format string (fmt) and up to three parameters (a,b,c)
 * this clobbers the original argv...
 */

/* VARARGS */
void setproctitle( const char *fmt, ... )
{
	static char *endargv = (char *)0;
	char	*s;
	int		i;
	char	buf[ 1024 ];
	va_list	ap;

	va_start(ap, fmt);

	buf[sizeof(buf) - 1] = '\0';
	vsnprintf( buf, sizeof(buf)-1, fmt, ap );

	va_end(ap);

	if ( endargv == (char *)0 ) {
		/* set pointer to end of original argv */
		endargv = Argv[ Argc-1 ] + strlen( Argv[ Argc-1 ] );
	}
	/* make ps print "([prog name])" */
	s = Argv[0];
	*s++ = '-';
	i = strlen( buf );
	if ( i > endargv - s - 2 ) {
		i = endargv - s - 2;
		buf[ i ] = '\0';
	}
	strcpy( s, buf );
	s += i;
	while ( s < endargv ) *s++ = ' ';
}
#endif /* NOSETPROCTITLE */
