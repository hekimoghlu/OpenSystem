/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#endif

/*
 * From:
 *  Id: find_interface.c,v 1.1 1995/08/14 16:08:39 wollman Exp
 */

#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "talk.h"

/*
 * Try to find the interface address that is used to route an IP
 * packet to a remote peer.
 */

int
get_iface(struct in_addr *dst, struct in_addr *iface)
{
	static struct sockaddr_in local;
	struct sockaddr_in remote;
	socklen_t namelen;
	int s, rv;

	memcpy(&remote.sin_addr, dst, sizeof remote.sin_addr);
	remote.sin_port = htons(60000);
	remote.sin_family = AF_INET;
	remote.sin_len = sizeof remote;

	local.sin_addr.s_addr = htonl(INADDR_ANY);
	local.sin_port = htons(60000);
	local.sin_family = AF_INET;
	local.sin_len = sizeof local;

	s = socket(PF_INET, SOCK_DGRAM, 0);
	if (s < 0)
		return -1;

	do {
		rv = bind(s, (struct sockaddr *)&local, sizeof local);
		local.sin_port = htons(ntohs(local.sin_port) + 1);
	} while(rv < 0 && errno == EADDRINUSE);

	if (rv < 0) {
		close(s);
		return -1;
	}

	do {
		rv = connect(s, (struct sockaddr *)&remote, sizeof remote);
		remote.sin_port = htons(ntohs(remote.sin_port) + 1);
	} while(rv < 0 && errno == EADDRINUSE);

	if (rv < 0) {
		close(s);
		return -1;
	}

	namelen = sizeof local;
	rv = getsockname(s, (struct sockaddr *)&local, &namelen);
	close(s);
	if (rv < 0)
		return -1;

	memcpy(iface, &local.sin_addr, sizeof local.sin_addr);
	return 0;
}
