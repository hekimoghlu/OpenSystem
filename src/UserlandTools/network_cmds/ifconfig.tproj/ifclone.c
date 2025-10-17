/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ifconfig.h"

static void
list_cloners(void)
{
	struct if_clonereq ifcr;
	char *cp, *buf;
	int idx;
	int s;

	s = socket(AF_INET, SOCK_DGRAM, 0);
	if (s == -1)
		err(1, "socket(AF_INET,SOCK_DGRAM)");

	memset(&ifcr, 0, sizeof(ifcr));

	if (ioctl(s, SIOCIFGCLONERS, &ifcr) < 0)
		err(1, "SIOCIFGCLONERS for count");

	buf = malloc(ifcr.ifcr_total * IFNAMSIZ);
	if (buf == NULL)
		err(1, "unable to allocate cloner name buffer");

	ifcr.ifcr_count = ifcr.ifcr_total;
	ifcr.ifcr_buffer = buf;

	if (ioctl(s, SIOCIFGCLONERS, &ifcr) < 0)
		err(1, "SIOCIFGCLONERS for names");

	/*
	 * In case some disappeared in the mean time, clamp it down.
	 */
	if (ifcr.ifcr_count > ifcr.ifcr_total)
		ifcr.ifcr_count = ifcr.ifcr_total;

	for (cp = buf, idx = 0; idx < ifcr.ifcr_count; idx++, cp += IFNAMSIZ) {
		if (idx > 0)
			putchar(' ');
		printf("%s", cp);
	}

	putchar('\n');
	free(buf);
}

static clone_callback_func *clone_cb = NULL;

void
clone_setcallback(clone_callback_func *p)
{
	if (clone_cb != NULL && clone_cb != p)
		errx(1, "conflicting device create parameters");
	clone_cb = p;
}

/*
 * Do the actual clone operation.  Any parameters must have been
 * setup by now.  If a callback has been setup to do the work
 * then defer to it; otherwise do a simple create operation with
 * no parameters.
 */
static void
ifclonecreate(int s, void *arg)
{
	struct ifreq ifr;

	memset(&ifr, 0, sizeof(ifr));
	(void) strlcpy(ifr.ifr_name, name, sizeof(ifr.ifr_name));
	if (clone_cb == NULL) {
#ifdef SIOCIFCREATE2
		/* NB: no parameters */
		if (ioctl(s, SIOCIFCREATE2, &ifr) < 0)
			err(1, "SIOCIFCREATE2");
#else
		if (ioctl(s, SIOCIFCREATE, &ifr) < 0)
			err(1, "SIOCIFCREATE");
#endif
	} else {
		clone_cb(s, &ifr);
	}

	/*
	 * If we get a different name back than we put in, print it.
	 */
	if (strncmp(name, ifr.ifr_name, sizeof(name)) != 0) {
		strlcpy(name, ifr.ifr_name, sizeof(name));
		printf("%s\n", name);
	}
}

static
DECL_CMD_FUNC(clone_create, arg, d)
{
	callback_register(ifclonecreate, NULL);
}

static
DECL_CMD_FUNC(clone_destroy, arg, d)
{
	(void) strlcpy(ifr.ifr_name, name, sizeof(ifr.ifr_name));
	if (ioctl(s, SIOCIFDESTROY, &ifr) < 0)
		err(1, "SIOCIFDESTROY");
}

static struct cmd clone_cmds[] = {
	DEF_CLONE_CMD("create",	0,	clone_create),
	DEF_CMD("destroy",	0,	clone_destroy),
	DEF_CLONE_CMD("plumb",	0,	clone_create),
	DEF_CMD("unplumb",	0,	clone_destroy),
};

static void
clone_Copt_cb(const char *optarg __unused)
{
	list_cloners();
	exit(0);
}
static struct option clone_Copt = { "C", "[-C]", clone_Copt_cb };

static __constructor void
clone_ctor(void)
{
#define	N(a)	(sizeof(a) / sizeof(a[0]))
	int i;

	for (i = 0; i < N(clone_cmds);  i++)
		cmd_register(&clone_cmds[i]);
	opt_register(&clone_Copt);
#undef N
}
