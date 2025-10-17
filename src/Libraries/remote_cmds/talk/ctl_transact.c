/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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
static const char sccsid[] = "@(#)ctl_transact.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* __APPLE__ */

#include <arpa/inet.h>

#include <errno.h>
#include <poll.h>

#include "talk.h"
#include "talk_ctl.h"

#define CTL_WAIT 2	/* time to wait for a response, in seconds */

/*
 * SOCKDGRAM is unreliable, so we must repeat messages if we have
 * not received an acknowledgement within a reasonable amount
 * of time
 */
void
ctl_transact(struct in_addr target, CTL_MSG lmsg, int type, CTL_RESPONSE *rp)
{
	struct pollfd pfd[1];
	int nready = 0, cc;

	lmsg.type = type;
	daemon_addr.sin_addr = target;
	daemon_addr.sin_port = daemon_port;
	pfd[0].fd = ctl_sockt;
	pfd[0].events = POLLIN;

	/*
	 * Keep sending the message until a response of
	 * the proper type is obtained.
	 */
	do {
		/* resend message until a response is obtained */
		do {
			cc = sendto(ctl_sockt, (char *)&lmsg, sizeof (lmsg), 0,
			    (struct sockaddr *)&daemon_addr,
			    sizeof (daemon_addr));
			if (cc != sizeof (lmsg)) {
				if (errno == EINTR)
					continue;
				p_error("Error on write to talk daemon");
			}
			nready = poll(pfd, 1, CTL_WAIT * 1000);
			if (nready < 0) {
				if (errno == EINTR)
					continue;
				p_error("Error waiting for daemon response");
			}
		} while (nready == 0);
		/*
		 * Keep reading while there are queued messages
		 * (this is not necessary, it just saves extra
		 * request/acknowledgements being sent)
		 */
		do {
			cc = recv(ctl_sockt, (char *)rp, sizeof (*rp), 0);
			if (cc < 0) {
				if (errno == EINTR)
					continue;
				p_error("Error on read from talk daemon");
			}
			nready = poll(pfd, 1, 0);
		} while (nready > 0 && (rp->vers != TALK_VERSION ||
		    rp->type != type));
	} while (rp->vers != TALK_VERSION || rp->type != type);
	rp->id_num = ntohl(rp->id_num);
	rp->addr.sa_family = ntohs(rp->addr.sa_family);
}
