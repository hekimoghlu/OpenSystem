/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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
static const char sccsid[] = "@(#)look_up.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* __APPLE__ */

#include <sys/types.h>
#include <sys/socket.h>
#include <protocols/talkd.h>

#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "talk_ctl.h"
#include "talk.h"

/*
 * See if the local daemon has an invitation for us.
 */
int
check_local(void)
{
	CTL_RESPONSE response;
	CTL_RESPONSE *rp = &response;
	struct sockaddr addr;

	/* the rest of msg was set up in get_names */
	/* copy new style sockaddr to old, swap family (short in old) */
	msg.ctl_addr = *(struct tsockaddr *)&ctl_addr;
	msg.ctl_addr.sa_family = htons(ctl_addr.sin_family);
	/* must be initiating a talk */
	if (!look_for_invite(rp))
		return (0);
	/*
	 * There was an invitation waiting for us,
	 * so connect with the other (hopefully waiting) party
	 */
	current_state = "Waiting to connect with caller";
	do {
		if (rp->addr.sa_family != AF_INET)
			p_error("Response uses invalid network address");
		(void)memcpy(&addr, &rp->addr.sa_family, sizeof(addr));
		addr.sa_family = rp->addr.sa_family;
		addr.sa_len = sizeof(addr);
		errno = 0;
		if (connect(sockt, &addr, sizeof(addr)) != -1)
			return (1);
	} while (errno == EINTR);
	if (errno == ECONNREFUSED) {
		/*
		 * The caller gave up, but his invitation somehow
		 * was not cleared. Clear it and initiate an
		 * invitation. (We know there are no newer invitations,
		 * the talkd works LIFO.)
		 */
		ctl_transact(his_machine_addr, msg, DELETE, rp);
		close(sockt);
		open_sockt();
		return (0);
	}
	p_error("Unable to connect with initiator");
	/*NOTREACHED*/
	return (0);
}

/*
 * Look for an invitation on 'machine'
 */
int
look_for_invite(CTL_RESPONSE *rp)
{
	current_state = "Checking for invitation on caller's machine";
	ctl_transact(his_machine_addr, msg, LOOK_UP, rp);
	/* the switch is for later options, such as multiple invitations */
	switch (rp->answer) {

	case SUCCESS:
		msg.id_num = htonl(rp->id_num);
		return (1);

	default:
		/* there wasn't an invitation waiting for us */
		return (0);
	}
}
