/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
static const char copyright[] =
"@(#) Copyright (c) 1988, 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif /* not lint */

#ifndef lint
static char sccsid[] = "@(#)chroot.c	8.1 (Berkeley) 6/9/93";
#endif /* not lint */
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>
#ifndef __APPLE__
#include <sys/procctl.h>
#endif

#include <ctype.h>
#include <err.h>
#include <grp.h>
#include <limits.h>
#include <paths.h>
#include <pwd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void usage(void);

int
main(int argc, char *argv[])
{
	struct group	*gp;
	struct passwd	*pw;
	char		*endp, *p, *user, *group, *grouplist;
	const char	*shell;
	gid_t		gid, *gidlist;
	uid_t		uid;
#ifdef __APPLE__
	/* arg, error are only used with nonprivileged. */
	int		ch, gids;
#else
	int		arg, ch, error, gids;
#endif
	long		ngroups_max;
#ifndef __APPLE__
	bool		nonprivileged;
#endif

	gid = 0;
	uid = 0;
	user = group = grouplist = NULL;
#ifdef __APPLE__
	while ((ch = getopt(argc, argv, "G:g:u:")) != -1) {
#else
	nonprivileged = false;
	while ((ch = getopt(argc, argv, "G:g:u:n")) != -1) {
#endif
		switch(ch) {
		case 'u':
			user = optarg;
			if (*user == '\0')
				usage();
			break;
		case 'g':
			group = optarg;
			if (*group == '\0')
				usage();
			break;
		case 'G':
			grouplist = optarg;
			if (*grouplist == '\0')
				usage();
			break;
#ifndef __APPLE__
		case 'n':
			nonprivileged = true;
			break;
#endif
		case '?':
		default:
			usage();
		}
	}
	argc -= optind;
	argv += optind;

	if (argc < 1)
		usage();

	if (group != NULL) {
		if (isdigit((unsigned char)*group)) {
			gid = (gid_t)strtoul(group, &endp, 0);
			if (*endp != '\0')
				goto getgroup;
		} else {
 getgroup:
			if ((gp = getgrnam(group)) != NULL)
				gid = gp->gr_gid;
			else
				errx(1, "no such group `%s'", group);
		}
	}

	ngroups_max = sysconf(_SC_NGROUPS_MAX) + 1;
	if ((gidlist = malloc(sizeof(gid_t) * ngroups_max)) == NULL)
		err(1, "malloc");
	for (gids = 0;
	    (p = strsep(&grouplist, ",")) != NULL && gids < ngroups_max; ) {
		if (*p == '\0')
			continue;

		if (isdigit((unsigned char)*p)) {
			gidlist[gids] = (gid_t)strtoul(p, &endp, 0);
			if (*endp != '\0')
				goto getglist;
		} else {
 getglist:
			if ((gp = getgrnam(p)) != NULL)
				gidlist[gids] = gp->gr_gid;
			else
				errx(1, "no such group `%s'", p);
		}
		gids++;
	}
	if (p != NULL && gids == ngroups_max)
		errx(1, "too many supplementary groups provided");

	if (user != NULL) {
		if (isdigit((unsigned char)*user)) {
			uid = (uid_t)strtoul(user, &endp, 0);
			if (*endp != '\0')
				goto getuser;
		} else {
 getuser:
			if ((pw = getpwnam(user)) != NULL)
				uid = pw->pw_uid;
			else
				errx(1, "no such user `%s'", user);
		}
	}

#ifndef __APPLE__
	if (nonprivileged) {
		arg = PROC_NO_NEW_PRIVS_ENABLE;
		error = procctl(P_PID, getpid(), PROC_NO_NEW_PRIVS_CTL, &arg);
		if (error != 0)
			err(1, "procctl");
	}
#endif

	if (chdir(argv[0]) == -1 || chroot(".") == -1)
		err(1, "%s", argv[0]);

	if (gids && setgroups(gids, gidlist) == -1)
		err(1, "setgroups");
	if (group && setgid(gid) == -1)
		err(1, "setgid");
	if (user && setuid(uid) == -1)
		err(1, "setuid");

	if (argv[1]) {
		execvp(argv[1], &argv[1]);
		err(1, "%s", argv[1]);
	}

	if (!(shell = getenv("SHELL")))
		shell = _PATH_BSHELL;
	execlp(shell, shell, "-i", (char *)NULL);
	err(1, "%s", shell);
	/* NOTREACHED */
}

static void
usage(void)
{
#ifdef __APPLE__
	(void)fprintf(stderr, "usage: chroot [-g group] [-G group,group,...] "
	    "[-u user] newroot [command]\n");
#else
	(void)fprintf(stderr, "usage: chroot [-g group] [-G group,group,...] "
	    "[-u user] [-n] newroot [command]\n");
#endif
	exit(1);
}
