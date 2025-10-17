/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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

__FBSDID("$FreeBSD$");

#ifndef lint
static const char sccsid[] = "@(#)get_addrs.c	8.1 (Berkeley) 6/6/93";
#endif

#include <err.h>
#include <netdb.h>
#include <string.h>
#include <unistd.h>

#include "talk.h"
#include "talk_ctl.h"

void
get_addrs(const char *my_machine_name __unused, const char *his_machine_name)
{
	struct hostent *hp;
	struct servent *sp;

	msg.pid = htonl(getpid());

	hp = gethostbyname(his_machine_name);
	if (hp == NULL)
		errx(1, "%s: %s", his_machine_name, hstrerror(h_errno));
	bcopy(hp->h_addr, (char *) &his_machine_addr, hp->h_length);
	if (get_iface(&his_machine_addr, &my_machine_addr) == -1)
		err(1, "failed to find my interface address");
	/* find the server's port */
	sp = getservbyname("ntalk", "udp");
	if (sp == NULL)
		errx(1, "ntalk/udp: service is not registered");
	daemon_port = sp->s_port;
}
