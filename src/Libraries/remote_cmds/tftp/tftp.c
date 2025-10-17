/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
static char sccsid[] = "@(#)tftp.c	8.1 (Berkeley) 6/6/93";
#endif /* not lint */
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

/* Many bug fixes are from Jim Guyton <guyton@rand-unix> */

/*
 * TFTP User Program -- Protocol Machines
 */
#include <sys/socket.h>
#include <sys/stat.h>

#include <netinet/in.h>

#include <arpa/tftp.h>

#include <assert.h>
#include <err.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>

#include "tftp.h"
#include "tftp-file.h"
#include "tftp-utils.h"
#include "tftp-io.h"
#include "tftp-transfer.h"
#include "tftp-options.h"

/*
 * Send the requested file.
 */
int
xmitfile(int peer, char *port, int fd, char *name, char *mode)
{
	struct tftphdr *rp;
	int n, i, ret = 0;
	uint16_t block;
	struct sockaddr_storage serv;	/* valid server port number */
	char recvbuffer[MAXPKTSIZE];
	struct tftp_stats tftp_stats;

	stats_init(&tftp_stats);

	memset(&serv, 0, sizeof(serv));
	rp = (struct tftphdr *)recvbuffer;

	if (port == NULL) {
		struct servent *se;
		se = getservbyname("tftp", "udp");
		assert(se != NULL);
		((struct sockaddr_in *)&peer_sock)->sin_port = se->s_port;
	} else
		((struct sockaddr_in *)&peer_sock)->sin_port =
		    htons(atoi(port));

	for (i = 0; i < 12; i++) {
		struct sockaddr_storage from;

		/* Tell the other side what we want to do */
		if (debug & DEBUG_SIMPLE)
			printf("Sending %s\n", name);

		n = send_wrq(peer, name, mode);
		if (n > 0) {
			printf("Cannot send WRQ packet\n");
			return -1;
		}

		/*
		 * The first packet we receive has the new destination port
		 * we have to send the next packets to.
		 */
		n = receive_packet(peer, recvbuffer,
		    MAXPKTSIZE, &from, timeoutpacket);

		/* We got some data! */
		if (n >= 0) {
			((struct sockaddr_in *)&peer_sock)->sin_port =
			    ((struct sockaddr_in *)&from)->sin_port;
			break;
		}

		/* This should be retried */
		if (n == RP_TIMEOUT) {
			printf("Try %d, didn't receive answer from remote.\n",
			    i + 1);
			continue;
		}

		/* Everything else is fatal */
		break;
	}
	if (i == 12) {
		printf("Transfer timed out.\n");
		return -1;
	}
	if (rp->th_opcode == ERROR) {
		printf("Got ERROR, aborted\n");
		return -1;
	}

	/*
	 * If the first packet is an OACK instead of an ACK packet,
	 * handle it different.
	 */
	if (rp->th_opcode == OACK) {
		if (!options_rfc_enabled) {
			printf("Got OACK while options are not enabled!\n");
			send_error(peer, EBADOP);
			return -1;
		}

		parse_options(peer, rp->th_stuff, n + 2);
	}

	if (read_init(fd, NULL, mode) < 0) {
		warn("read_init()");
		return -1;
	}

	block = 1;
	if (tftp_send(peer, &block, &tftp_stats) != 0)
		ret = -1;

	read_close();
	if (tftp_stats.amount > 0)
		printstats("Sent", verbose, &tftp_stats);
	return ret;
}

/*
 * Receive a file.
 */
int
recvfile(int peer, char *port, int fd, char *name, char *mode)
{
	struct tftphdr *rp;
	uint16_t block;
	char recvbuffer[MAXPKTSIZE];
	int n, i, ret = 0;
	struct tftp_stats tftp_stats;

	stats_init(&tftp_stats);

	rp = (struct tftphdr *)recvbuffer;

	if (port == NULL) {
		struct servent *se;
		se = getservbyname("tftp", "udp");
		assert(se != NULL);
		((struct sockaddr_in *)&peer_sock)->sin_port = se->s_port;
	} else
		((struct sockaddr_in *)&peer_sock)->sin_port =
		    htons(atoi(port));

	for (i = 0; i < 12; i++) {
		struct sockaddr_storage from;

		/* Tell the other side what we want to do */
		if (debug & DEBUG_SIMPLE)
			printf("Requesting %s\n", name);

		n = send_rrq(peer, name, mode);
		if (n > 0) {
			printf("Cannot send RRQ packet\n");
			return -1;
		}

		/*
		 * The first packet we receive has the new destination port
		 * we have to send the next packets to.
		 */
		n = receive_packet(peer, recvbuffer,
		    MAXPKTSIZE, &from, timeoutpacket);

		/* We got something useful! */
		if (n >= 0) {
			((struct sockaddr_in *)&peer_sock)->sin_port =
			    ((struct sockaddr_in *)&from)->sin_port;
			break;
		}

		/* We should retry if this happens */
		if (n == RP_TIMEOUT) {
			printf("Try %d, didn't receive answer from remote.\n",
			    i + 1);
			continue;
		}

		/* Otherwise it is a fatal error */
		break;
	}
	if (i == 12) {
		printf("Transfer timed out.\n");
		return -1;
	}
	if (rp->th_opcode == ERROR) {
		tftp_log(LOG_ERR, "Error code %d: %s", rp->th_code, rp->th_msg);
		return -1;
	}

	if (write_init(fd, NULL, mode) < 0) {
		warn("write_init");
		return -1;
	}

	/*
	 * If the first packet is an OACK packet instead of an DATA packet,
	 * handle it different.
	 */
	if (rp->th_opcode == OACK) {
		if (!options_rfc_enabled) {
			printf("Got OACK while options are not enabled!\n");
			send_error(peer, EBADOP);
			return -1;
		}

		parse_options(peer, rp->th_stuff, n + 2);

		n = send_ack(peer, 0);
		if (n > 0) {
			printf("Cannot send ACK on OACK.\n");
			return -1;
		}
		block = 0;
		if (tftp_receive(peer, &block, &tftp_stats, NULL, 0) != 0)
			ret = -1;
	} else {
		block = 1;
		if (tftp_receive(peer, &block, &tftp_stats, rp, n) != 0)
			ret = -1;
	}

	if (tftp_stats.amount > 0)
		printstats("Received", verbose, &tftp_stats);
	return ret;
}
