/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
#if 0
static char sccsid[] = "@(#)ls.c	8.1 (Berkeley) 6/6/93";
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/param.h>
#include <sys/stat.h>

#include <err.h>
#include <errno.h>
#include <fts.h>
#include <grp.h>
#include <inttypes.h>
#include <langinfo.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#undef MAXLOGNAME
#define MAXLOGNAME 17
#endif /* __APPLE__ */

#include "find.h"

/* Derived from the print routines in the ls(1) source code. */

static void printlink(char *);
static void printtime(time_t);

void
printlong(char *name, char *accpath, struct stat *sb)
{
	char modep[15];

	(void)printf("%6ju %8"PRId64" ", (uintmax_t)sb->st_ino, sb->st_blocks);
	(void)strmode(sb->st_mode, modep);
	(void)printf("%s %3ju %-*s %-*s ", modep, (uintmax_t)sb->st_nlink,
	    MAXLOGNAME - 1,
	    user_from_uid(sb->st_uid, 0), MAXLOGNAME - 1,
	    group_from_gid(sb->st_gid, 0));

	if (S_ISCHR(sb->st_mode) || S_ISBLK(sb->st_mode))
		(void)printf("%#8jx ", (uintmax_t)sb->st_rdev);
	else
		(void)printf("%8"PRId64" ", sb->st_size);
	printtime(sb->st_mtime);
	(void)printf("%s", name);
	if (S_ISLNK(sb->st_mode))
		printlink(accpath);
	(void)putchar('\n');
}

static void
printtime(time_t ftime)
{
	char longstring[80];
	static time_t lnow;
	const char *format;
	static int d_first = -1;
	struct tm *tm;

#ifdef D_MD_ORDER
	if (d_first < 0)
		d_first = (*nl_langinfo(D_MD_ORDER) == 'd');
#endif
	if (lnow == 0)
		lnow = time(NULL);

#define	SIXMONTHS	((365 / 2) * 86400)
	if (ftime + SIXMONTHS > lnow && ftime < lnow + SIXMONTHS)
		/* mmm dd hh:mm || dd mmm hh:mm */
		format = d_first ? "%e %b %R " : "%b %e %R ";
	else
		/* mmm dd  yyyy || dd mmm  yyyy */
		format = d_first ? "%e %b  %Y " : "%b %e  %Y ";
	if ((tm = localtime(&ftime)) != NULL)
		strftime(longstring, sizeof(longstring), format, tm);
	else
		strlcpy(longstring, "bad date val ", sizeof(longstring));
	fputs(longstring, stdout);
}

static void
printlink(char *name)
{
	ssize_t lnklen;
	char path[MAXPATHLEN];

	if ((lnklen = readlink(name, path, MAXPATHLEN - 1)) == -1) {
		warn("%s", name);
		return;
	}
	path[lnklen] = '\0';
	(void)printf(" -> %s", path);
}
