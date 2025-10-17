/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#include "telnet_locl.h"

RCSID("$Id$");

Ring		netoring, netiring;
size_t		netobufsize = 64*1024;
size_t		netibufsize = 64*1024;

/*
 * Initialize internal network data structures.
 */

void
init_network(void)
{
    void *obuf, *ibuf;

    if ((obuf = malloc(netobufsize)) == NULL)
	exit(1);
    if ((ibuf = malloc(netibufsize)) == NULL)
	exit(1);

    if (ring_init(&netoring, obuf, netobufsize) != 1) {
	exit(1);
    }
    if (ring_init(&netiring, ibuf, netibufsize) != 1) {
	exit(1);
    }
    NetTrace = stdout;
}


/*
 * Check to see if any out-of-band data exists on a socket (for
 * Telnet "synch" processing).
 */

int
stilloob(void)
{
    static struct timeval timeout = { 0 };
    fd_set	excepts;
    int value;

    do {
	FD_ZERO(&excepts);
	if (net >= FD_SETSIZE)
	    errx (1, "fd too large");
	FD_SET(net, &excepts);
	value = select(net+1, 0, 0, &excepts, &timeout);
    } while ((value == -1) && (errno == EINTR));

    if (value < 0) {
	perror("select");
	quit();
	/* NOTREACHED */
    }
    if (FD_ISSET(net, &excepts)) {
	return 1;
    } else {
	return 0;
    }
}


/*
 *  setneturg()
 *
 *	Sets "neturg" to the current location.
 */

void
setneturg(void)
{
    ring_mark(&netoring);
}


/*
 *  netflush
 *		Send as much data as possible to the network,
 *	handling requests for urgent data.
 *
 *		The return value indicates whether we did any
 *	useful work.
 */


int
netflush(void)
{
    int n, n1;

#if	defined(ENCRYPTION)
    if (encrypt_output)
	ring_encrypt(&netoring, encrypt_output);
#endif
    if ((n1 = n = ring_full_consecutive(&netoring)) > 0) {
	if (!ring_at_mark(&netoring)) {
	    n = send(net, (char *)netoring.consume, n, 0); /* normal write */
	} else {
	    /*
	     * In 4.2 (and 4.3) systems, there is some question about
	     * what byte in a sendOOB operation is the "OOB" data.
	     * To make ourselves compatible, we only send ONE byte
	     * out of band, the one WE THINK should be OOB (though
	     * we really have more the TCP philosophy of urgent data
	     * rather than the Unix philosophy of OOB data).
	     */
	    n = send(net, (char *)netoring.consume, 1, MSG_OOB);/* URGENT data */
	}
    }
    if (n < 0) {
	if (errno != ENOBUFS && errno != EWOULDBLOCK) {
	    setcommandmode();
	    perror(hostname);
	    NetClose(net);
	    ring_clear_mark(&netoring);
	    longjmp(peerdied, -1);
	    /*NOTREACHED*/
	}
	n = 0;
    }
    if (netdata && n) {
	Dump('>', netoring.consume, n);
    }
    if (n) {
	ring_consumed(&netoring, n);
	/*
	 * If we sent all, and more to send, then recurse to pick
	 * up the other half.
	 */
	if ((n1 == n) && ring_full_consecutive(&netoring)) {
	    netflush();
	}
	return 1;
    } else {
	return 0;
    }
}
