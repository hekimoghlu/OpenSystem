/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
static char sccsid[] = "@(#)psignal.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/psignal.c,v 1.8 2007/01/09 00:27:54 imp Exp $");

/*
 * Print the name of the signal indicated
 * along with the supplied message.
 */
#include "namespace.h"
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include "un-namespace.h"

void
psignal(sig, s)
	unsigned int sig;
	const char *s;
{
	const char *c;

	if (sig < NSIG)
		c = sys_siglist[sig];
	else
		c = "Unknown signal";
	if (s != NULL && *s != '\0') {
		(void)_write(STDERR_FILENO, s, strlen(s));
		(void)_write(STDERR_FILENO, ": ", 2);
	}
	(void)_write(STDERR_FILENO, c, strlen(c));
	(void)_write(STDERR_FILENO, "\n", 1);
}
