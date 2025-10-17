/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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
static char sccsid[] = "@(#)gets.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdio/gets.c,v 1.17 2007/01/09 00:28:06 imp Exp $");

#include "namespace.h"
#include <unistd.h>
#include <stdio.h>
#include <sys/cdefs.h>
#include "un-namespace.h"
#include "libc_private.h"
#include "local.h"

__warn_references(gets, "warning: this program uses gets(), which is unsafe.");

char *
gets(char *buf)
{
	int c;
	char *s;
	static int warned;
	static const char w[] =
	    "warning: this program uses gets(), which is unsafe.\n";

	FLOCKFILE(stdin);
	ORIENT(stdin, -1);
	if (!warned) {
		(void) _write(STDERR_FILENO, w, sizeof(w) - 1);
		warned = 1;
	}
	for (s = buf; (c = __sgetc(stdin)) != '\n';)
		if (c == EOF)
			if (s == buf) {
				FUNLOCKFILE(stdin);
				return (NULL);
			} else
				break;
		else
			*s++ = c;
	*s = 0;
	FUNLOCKFILE(stdin);
	return (buf);
}
