/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
static char sccsid[] = "@(#)v7.local.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

/*
 * Mail -- a mail program
 *
 * Version 7
 *
 * Local routines that are installation dependent.
 */

#include "rcv.h"
#include <fcntl.h>
#include "extern.h"

/*
 * Locate the user's mailbox file (ie, the place where new, unread
 * mail is queued).
 */
void
findmail(char *user, char *buf, int buflen)
{
	char *tmp = getenv("MAIL");

	if (tmp == NULL)
		(void)snprintf(buf, buflen, "%s/%s", _PATH_MAILDIR, user);
	else
		(void)strlcpy(buf, tmp, buflen);
}

/*
 * Get rid of the queued mail.
 */
void
demail(void)
{
	int fd;

	if (value("keep") != NULL || rm(mailname) < 0)
		if ((fd = open(mailname, O_CREAT | O_TRUNC | O_WRONLY, 0600)) >=
		    0)
			(void)close(fd);
}

/*
 * Discover user login name.
 */
char *
username(void)
{
	char *np;
	uid_t uid;

	if ((np = getenv("USER")) != NULL)
		return (np);
	if ((np = getenv("LOGNAME")) != NULL)
		return (np);
	if ((np = getname(uid = getuid())) != NULL)
		return (np);
	printf("Cannot associate a name with uid %u\n", (unsigned)uid);
	return (NULL);
}
