/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#include <sys/cdefs.h>
__SCCSID("@(#)psignal.c	8.1 (Berkeley) 6/4/93");
__FBSDID("$FreeBSD$");

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
psignal(int sig, const char *s)
{
	const char *c;

	if (sig >= 0 && sig < NSIG)
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
