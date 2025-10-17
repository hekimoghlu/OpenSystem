/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)temp.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "rcv.h"
#include "extern.h"

/*
 * Mail -- a mail program
 *
 * Give names to all the temporary files that we will need.
 */

char	*tmpdir;

void
tinit(void)
{
	char *cp;

	if ((tmpdir = getenv("TMPDIR")) == NULL || *tmpdir == '\0')
		tmpdir = _PATH_TMP;
	if ((tmpdir = strdup(tmpdir)) == NULL)
		errx(1, "Out of memory");
	/* Strip trailing '/' if necessary */
	cp = tmpdir + strlen(tmpdir) - 1;
	while (cp > tmpdir && *cp == '/') {
		*cp = '\0';
		cp--;
	}

	/*
	 * It's okay to call savestr in here because main will
	 * do a spreserve() after us.
	 */
	if (myname != NULL) {
		if (getuserid(myname) < 0)
			errx(1, "\"%s\" is not a user of this system", myname);
	} else {
		if ((cp = username()) == NULL) {
			myname = "ubluit";
			if (rcvmode)
				errx(1, "Who are you!?");
		} else
			myname = savestr(cp);
	}
	if ((cp = getenv("HOME")) == NULL || *cp == '\0' ||
	    strlen(cp) >= PATHSIZE)
		homedir = NULL;
	else
		homedir = savestr(cp);
	if (debug)
		fprintf(stderr, "user = %s, homedir = %s\n", myname,
		    homedir ? homedir : "NONE");
}
