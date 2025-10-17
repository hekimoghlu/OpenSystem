/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
static char sccsid[] = "@(#)mail.c	8.2 (Berkeley) 5/4/95";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: head/bin/sh/mail.c 314436 2017-02-28 23:42:47Z imp $");

/*
 * Routines to check for mail.  (Perhaps make part of main.c?)
 */

#include "shell.h"
#include "exec.h"	/* defines padvance() */
#include "mail.h"
#include "var.h"
#include "output.h"
#include "memalloc.h"
#include "error.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>


#define MAXMBOXES 10


static int nmboxes;			/* number of mailboxes */
static time_t mailtime[MAXMBOXES];	/* times of mailboxes */



/*
 * Print appropriate message(s) if mail has arrived.  If the argument is
 * nozero, then the value of MAIL has changed, so we just update the
 * values.
 */

void
chkmail(int silent)
{
	int i;
	const char *mpath;
	char *p;
	char *q;
	struct stackmark smark;
	struct stat statb;

	if (silent)
		nmboxes = 10;
	if (nmboxes == 0)
		return;
	setstackmark(&smark);
	mpath = mpathset()? mpathval() : mailval();
	for (i = 0 ; i < nmboxes ; i++) {
		p = padvance(&mpath, "");
		if (p == NULL)
			break;
		if (*p == '\0')
			continue;
		for (q = p ; *q ; q++);
		if (q[-1] != '/')
			abort();
		q[-1] = '\0';			/* delete trailing '/' */
#ifdef notdef /* this is what the System V shell claims to do (it lies) */
		if (stat(p, &statb) < 0)
			statb.st_mtime = 0;
		if (statb.st_mtime > mailtime[i] && ! silent) {
			out2str(pathopt? pathopt : "you have mail");
			out2c('\n');
		}
		mailtime[i] = statb.st_mtime;
#else /* this is what it should do */
		if (stat(p, &statb) < 0)
			statb.st_size = 0;
		if (statb.st_size > mailtime[i] && ! silent) {
			out2str(pathopt? pathopt : "you have mail");
			out2c('\n');
		}
		mailtime[i] = statb.st_size;
#endif
	}
	nmboxes = i;
	popstackmark(&smark);
}
