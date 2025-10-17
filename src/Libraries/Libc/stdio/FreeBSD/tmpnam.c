/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)tmpnam.c	8.3 (Berkeley) 3/28/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/tmpnam.c,v 1.6 2007/01/09 00:28:07 imp Exp $");

#include <sys/types.h>

#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

#include "libc_hooks_impl.h"

__warn_references(tmpnam,
    "warning: tmpnam() possibly used unsafely; consider using mkstemp()");

extern char *_mktemp(char *);

static char *tmpnam_buf = NULL;
static pthread_once_t tmpnam_buf_control = PTHREAD_ONCE_INIT;

static void tmpnam_buf_allocate(void)
{
	tmpnam_buf = malloc(L_tmpnam);
}

char *
tmpnam(char *s)
{
	static u_long tmpcount;

	if (s == NULL) {
		if (pthread_once(&tmpnam_buf_control, tmpnam_buf_allocate)
			|| !tmpnam_buf) {
			return NULL;
		}
		s = tmpnam_buf;
	}

	libc_hooks_will_write(s, L_tmpnam);

	(void)snprintf(s, L_tmpnam, "%stmp.%lu.XXXXXX", P_tmpdir, tmpcount);
	++tmpcount;
	return (_mktemp(s));
}
