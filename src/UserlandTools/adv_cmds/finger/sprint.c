/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
#ifndef lint
static char sccsid[] = "@(#)sprint.c	8.3 (Berkeley) 4/28/95";
#endif
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/param.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <db.h>
#include <err.h>
#include <langinfo.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#ifdef __APPLE__
#include <sys/time.h>
#else
#include <time.h>
#endif	/* __APPLE */
#include <utmpx.h>
#include "finger.h"

static void	  stimeprint(WHERE *);

void
sflag_print(void)
{
	PERSON *pn;
	WHERE *w;
	int sflag, r, namelen;
	char p[80];
	PERSON *tmp;
	DBT data, key;
	struct tm *lc;

	if (d_first < 0)
		d_first = (*nl_langinfo(D_MD_ORDER) == 'd');
	/*
	 * short format --
	 *	login name
	 *	real name
	 *	terminal name (the XX of ttyXX)
	 *	if terminal writeable (add an '*' to the terminal name
	 *		if not)
	 *	if logged in show idle time and day logged in, else
	 *		show last login date and time.
	 *		If > 6 months, show year instead of time.
	 *	if (-o)
	 *		office location
	 *		office phone
	 *	else
	 *		remote host
	 */
#define	MAXREALNAME	16
#define MAXHOSTNAME     17      /* in reality, hosts are never longer than 16 */
#ifdef __APPLE__
/* 
 * On APPLE MAXLOGNAME is 256 characters which is impractical to print,
 * truncate to 16 for printing.
 */
#define SHORTLOGNAME	16 	
	(void)printf("%-*s %-*s%s %s\n", SHORTLOGNAME, "Login", MAXREALNAME,
#else
	(void)printf("%-*s %-*s%s %s\n", MAXLOGNAME, "Login", MAXREALNAME,
#endif	/* __APPLE__ */
	    "Name", " TTY      Idle  Login  Time  ", (gflag) ? "" :
	    oflag ? "Office  Phone" : "Where");

	for (sflag = R_FIRST;; sflag = R_NEXT) {
		r = (*db->seq)(db, &key, &data, sflag);
		if (r == -1)
			err(1, "db seq");
		if (r == 1)
			break;
		memmove(&tmp, data.data, sizeof tmp);
		pn = tmp;

		for (w = pn->whead; w != NULL; w = w->next) {
			namelen = MAXREALNAME;
			if (w->info == LOGGEDIN && !w->writable)
				--namelen;	/* leave space before `*' */
#ifdef __APPLE__
			(void)printf("%-*.*s %-*.*s", SHORTLOGNAME, SHORTLOGNAME,
#else
			(void)printf("%-*.*s %-*.*s", MAXLOGNAME, MAXLOGNAME,
#endif /* __APPLE__ */
				pn->name, MAXREALNAME, namelen,
				pn->realname ? pn->realname : "");
			if (!w->loginat) {
				(void)printf("  *     *   No logins   ");
				goto office;
			}
			(void)putchar(w->info == LOGGEDIN && !w->writable ?
			    '*' : ' ');
			if (*w->tty)
				(void)printf("%-7.7s ",
					     (strncmp(w->tty, "tty", 3)
					      && strncmp(w->tty, "cua", 3))
					     ? w->tty : w->tty + 3);
			else
				(void)printf("        ");
			if (w->info == LOGGEDIN) {
				stimeprint(w);
				(void)printf("  ");
			} else
				(void)printf("    *  ");
			lc = localtime(&w->loginat);
#define SECSPERDAY 86400
#define DAYSPERWEEK 7
#define DAYSPERNYEAR 365
			if (now - w->loginat < SECSPERDAY * (DAYSPERWEEK - 1)) {
				(void)strftime(p, sizeof(p), "%a", lc);
			} else {
				(void)strftime(p, sizeof(p),
					     d_first ? "%e %b" : "%b %e", lc);
			}
			(void)printf("%-6.6s", p);
			if (now - w->loginat >= SECSPERDAY * DAYSPERNYEAR / 2) {
				(void)strftime(p, sizeof(p), "%Y", lc);
			} else {
				(void)strftime(p, sizeof(p), "%R", lc);
			}
			(void)printf(" %-5.5s", p);
office:
			if (gflag)
				goto no_gecos;
			if (oflag) {
				if (pn->office)
					(void)printf(" %-7.7s", pn->office);
				else if (pn->officephone)
					(void)printf(" %-7.7s", " ");
				if (pn->officephone)
					(void)printf(" %-.15s",
					    prphone(pn->officephone));
			} else
				(void)printf(" %.*s", MAXHOSTNAME, w->host);
no_gecos:
			putchar('\n');
		}
	}
}

static void
stimeprint(WHERE *w)
{
	struct tm *delta;

	if (w->idletime == -1) {
		(void)printf("     ");
		return;
	}

	delta = gmtime(&w->idletime);
	if (!delta->tm_yday)
		if (!delta->tm_hour)
			if (!delta->tm_min)
				(void)printf("     ");
			else
				(void)printf("%5d", delta->tm_min);
		else
			(void)printf("%2d:%02d",
			    delta->tm_hour, delta->tm_min);
	else
		(void)printf("%4dd", delta->tm_yday);
}
