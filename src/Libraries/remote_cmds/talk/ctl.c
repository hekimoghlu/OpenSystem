/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
static const char sccsid[] = "@(#)ctl.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* __APPLE__ */

/*
 * This file handles haggling with the various talk daemons to
 * get a socket to talk to. sockt is opened and connected in
 * the progress
 */

#include <sys/types.h>
#include <sys/socket.h>

#include <string.h>

#include "talk.h"
#include "talk_ctl.h"

struct	sockaddr_in daemon_addr = { .sin_len = sizeof(daemon_addr), .sin_family = AF_INET };
struct	sockaddr_in ctl_addr = { .sin_len = sizeof(ctl_addr), .sin_family = AF_INET };
struct	sockaddr_in my_addr = { .sin_len = sizeof(my_addr), .sin_family = AF_INET };

	/* inet addresses of the two machines */
struct	in_addr my_machine_addr;
struct	in_addr his_machine_addr;

u_short daemon_port;	/* port number of the talk daemon */

int	ctl_sockt;
int	sockt;
int	invitation_waiting = 0;

CTL_MSG msg;

void
open_sockt(void)
{
	socklen_t length;

	(void)memset(&my_addr, 0, sizeof(my_addr));
	my_addr.sin_family = AF_INET;
	my_addr.sin_len = sizeof(my_addr);
	my_addr.sin_addr = my_machine_addr;
	my_addr.sin_port = 0;
	sockt = socket(AF_INET, SOCK_STREAM, 0);
	if (sockt == -1)
		p_error("Bad socket");
	if (bind(sockt, (struct sockaddr *)&my_addr, sizeof(my_addr)) != 0)
		p_error("Binding local socket");
	length = sizeof(my_addr);
	if (getsockname(sockt, (struct sockaddr *)&my_addr, &length) == -1)
		p_error("Bad address for socket");
}

/* open the ctl socket */
void
open_ctl(void)
{
	socklen_t length;

	(void)memset(&ctl_addr, 0, sizeof(ctl_addr));
	ctl_addr.sin_family = AF_INET;
	ctl_addr.sin_len = sizeof(my_addr);
	ctl_addr.sin_port = 0;
	ctl_addr.sin_addr = my_machine_addr;
	ctl_sockt = socket(AF_INET, SOCK_DGRAM, 0);
	if (ctl_sockt == -1)
		p_error("Bad socket");
	if (bind(ctl_sockt,
	    (struct sockaddr *)&ctl_addr, sizeof(ctl_addr)) != 0)
		p_error("Couldn't bind to control socket");
	length = sizeof(ctl_addr);
	if (getsockname(ctl_sockt,
	    (struct sockaddr *)&ctl_addr, &length) == -1)
		p_error("Bad address for ctl socket");
}

/* print_addr is a debug print routine */
void
print_addr(struct sockaddr_in addr)
{
	int i;

	printf("addr = %lx, port = %o, family = %o zero = ",
		(u_long)addr.sin_addr.s_addr, addr.sin_port, addr.sin_family);
	for (i = 0; i<8;i++)
	printf("%o ", (int)addr.sin_zero[i]);
	putchar('\n');
}
