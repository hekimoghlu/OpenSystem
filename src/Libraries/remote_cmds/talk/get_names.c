/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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

#ifndef __APPLE__
__FBSDID("$FreeBSD$");

#ifndef lint
static const char sccsid[] = "@(#)get_names.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* __APPLE__ */

#include <sys/param.h>

#include <err.h>
#include <pwd.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "talk.h"

extern	CTL_MSG msg;

static void
usage(void)
{
	fprintf(stderr, "usage: talk person [ttyname]\n");
	exit(1);
}

/*
 * Determine the local and remote user, tty, and machines
 */
void
get_names(int argc, char *argv[])
{
	char hostname[MAXHOSTNAMELEN];
	char *his_name, *my_name;
	const char *my_machine_name, *his_machine_name;
	const char *his_tty;
	char *cp;

	if (argc < 2 )
		usage();
	if (!isatty(0))
		errx(1, "standard input must be a tty, not a pipe or a file");
	if ((my_name = getlogin()) == NULL) {
		struct passwd *pw;

		if ((pw = getpwuid(getuid())) == NULL)
			errx(1, "you don't exist. Go away");
		my_name = pw->pw_name;
	}
	gethostname(hostname, sizeof (hostname));
	my_machine_name = hostname;
	/* check for, and strip out, the machine name of the target */
	cp = argv[1] + strcspn(argv[1], "@:!");
	if (*cp == '\0') {
		/* this is a local to local talk */
		his_name = argv[1];
		my_machine_name = his_machine_name = "localhost";
	} else {
		if (*cp++ == '@') {
			/* user@host */
			his_name = argv[1];
			his_machine_name = cp;
		} else {
			/* host!user or host:user */
			his_name = cp;
			his_machine_name = argv[1];
		}
		*--cp = '\0';
	}
	if (argc > 2)
		his_tty = argv[2];	/* tty name is arg 2 */
	else
		his_tty = "";
	get_addrs(my_machine_name, his_machine_name);
	/*
	 * Initialize the message template.
	 */
	msg.vers = TALK_VERSION;
	msg.addr.sa_family = htons(AF_INET);
	msg.ctl_addr.sa_family = htons(AF_INET);
	msg.id_num = htonl(0);
	strlcpy(msg.l_name, my_name, NAME_SIZE);
	strlcpy(msg.r_name, his_name, NAME_SIZE);
	strlcpy(msg.r_tty, his_tty, TTY_SIZE);
}
