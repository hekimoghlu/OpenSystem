/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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
/* System interfaces. */

#include <sys_defs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>

/* Utility library. */

#include "mymalloc.h"
#include "msg.h"
#include "iostuff.h"
#include "host_port.h"
#include "sane_connect.h"
#include "connect.h"
#include "timed_connect.h"
#include "myaddrinfo.h"
#include "sock_addr.h"
#include "inet_proto.h"

static int inet_connect_one(struct addrinfo *, int, int);

/* inet_connect - connect to TCP listener */

int     inet_connect(const char *addr, int block_mode, int timeout)
{
    char   *buf;
    char   *host;
    char   *port;
    const char *parse_err;
    struct addrinfo *res;
    struct addrinfo *res0;
    int     aierr;
    int     sock;
    MAI_HOSTADDR_STR hostaddr;
    INET_PROTO_INFO *proto_info;
    int     found;

    /*
     * Translate address information to internal form. No host defaults to
     * the local host.
     */
    buf = mystrdup(addr);
    if ((parse_err = host_port(buf, &host, "localhost", &port, (char *) 0)) != 0)
	msg_fatal("%s: %s", addr, parse_err);
    if ((aierr = hostname_to_sockaddr(host, port, SOCK_STREAM, &res0)) != 0)
	msg_fatal("host/service %s/%s not found: %s",
		  host, port, MAI_STRERROR(aierr));
    myfree(buf);

    proto_info = inet_proto_info();
    for (sock = -1, found = 0, res = res0; res != 0; res = res->ai_next) {

	/*
	 * Safety net.
	 */
	if (strchr((char *) proto_info->sa_family_list, res->ai_family) == 0) {
	    msg_info("skipping address family %d for host %s",
		     res->ai_family, host);
	    continue;
	}
	found++;

	/*
	 * In case of multiple addresses, show what address we're trying now.
	 */
	if (msg_verbose) {
	    SOCKADDR_TO_HOSTADDR(res->ai_addr, res->ai_addrlen,
				 &hostaddr, (MAI_SERVPORT_STR *) 0, 0);
	    msg_info("trying... [%s]", hostaddr.buf);
	}
	if ((sock = inet_connect_one(res, block_mode, timeout)) < 0) {
	    if (msg_verbose)
		msg_info("%m");
	} else
	    break;
    }
    if (found == 0)
	msg_fatal("host not found: %s", addr);
    freeaddrinfo(res0);
    return (sock);
}

/* inet_connect_one - try to connect to one address */

static int inet_connect_one(struct addrinfo * res, int block_mode, int timeout)
{
    int     sock;

    /*
     * Create a client socket.
     */
    sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock < 0)
	return (-1);

    /*
     * Window scaling workaround.
     */
    if (inet_windowsize > 0)
	set_inet_windowsize(sock, inet_windowsize);

    /*
     * Timed connect.
     */
    if (timeout > 0) {
	non_blocking(sock, NON_BLOCKING);
	if (timed_connect(sock, res->ai_addr, res->ai_addrlen, timeout) < 0) {
	    close(sock);
	    return (-1);
	}
	if (block_mode != NON_BLOCKING)
	    non_blocking(sock, block_mode);
	return (sock);
    }

    /*
     * Maybe block until connected.
     */
    else {
	non_blocking(sock, block_mode);
	if (sane_connect(sock, res->ai_addr, res->ai_addrlen) < 0
	    && errno != EINPROGRESS) {
	    close(sock);
	    return (-1);
	}
	return (sock);
    }
}
