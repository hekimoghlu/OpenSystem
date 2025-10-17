/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
/* System libraries. */

#include <sys_defs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#ifndef MAXHOSTNAMELEN
#include <sys/param.h>
#endif
#include <errno.h>
#include <string.h>
#include <unistd.h>

/* Utility library. */

#include "mymalloc.h"
#include "msg.h"
#include "host_port.h"
#include "iostuff.h"
#include "listen.h"
#include "sane_accept.h"
#include "myaddrinfo.h"
#include "sock_addr.h"
#include "inet_proto.h"

/* inet_listen - create TCP listener */

int     inet_listen(const char *addr, int backlog, int block_mode)
{
    struct addrinfo *res;
    struct addrinfo *res0;
    int     aierr;
    int     sock;
    int     on = 1;
    char   *buf;
    char   *host;
    char   *port;
    const char *parse_err;
    MAI_HOSTADDR_STR hostaddr;
    MAI_SERVPORT_STR portnum;
    INET_PROTO_INFO *proto_info;

    /*
     * Translate address information to internal form.
     */
    buf = mystrdup(addr);
    if ((parse_err = host_port(buf, &host, "", &port, (char *) 0)) != 0)
	msg_fatal("%s: %s", addr, parse_err);
    if (*host == 0)
	host = 0;
    if ((aierr = hostname_to_sockaddr(host, port, SOCK_STREAM, &res0)) != 0)
	msg_fatal("%s: %s", addr, MAI_STRERROR(aierr));
    myfree(buf);
    /* No early returns or res0 leaks. */

    proto_info = inet_proto_info();
    for (res = res0; /* see below */ ; res = res->ai_next) {

	/*
	 * No usable address found.
	 */
	if (res == 0)
	    msg_fatal("%s: host found but no usable address", addr);

	/*
	 * Safety net.
	 */
	if (strchr((char *) proto_info->sa_family_list, res->ai_family) != 0)
	    break;

	msg_info("skipping address family %d for %s", res->ai_family, addr);
    }

    /*
     * Show what address we're trying.
     */
    if (msg_verbose) {
	SOCKADDR_TO_HOSTADDR(res->ai_addr, res->ai_addrlen,
			     &hostaddr, &portnum, 0);
	msg_info("trying... [%s]:%s", hostaddr.buf, portnum.buf);
    }

    /*
     * Create a listener socket.
     */
    if ((sock = socket(res->ai_family, res->ai_socktype, 0)) < 0)
	msg_fatal("socket: %m");
#ifdef HAS_IPV6
#if defined(IPV6_V6ONLY) && !defined(BROKEN_AI_PASSIVE_NULL_HOST)
    if (res->ai_family == AF_INET6
	&& setsockopt(sock, IPPROTO_IPV6, IPV6_V6ONLY,
		      (void *) &on, sizeof(on)) < 0)
	msg_fatal("setsockopt(IPV6_V6ONLY): %m");
#endif
#endif
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
		   (void *) &on, sizeof(on)) < 0)
	msg_fatal("setsockopt(SO_REUSEADDR): %m");
    if (bind(sock, res->ai_addr, res->ai_addrlen) < 0) {
	SOCKADDR_TO_HOSTADDR(res->ai_addr, res->ai_addrlen,
			     &hostaddr, &portnum, 0);
	msg_fatal("bind %s port %s: %m", hostaddr.buf, portnum.buf);
    }
    freeaddrinfo(res0);
    non_blocking(sock, block_mode);
    if (inet_windowsize > 0)
	set_inet_windowsize(sock, inet_windowsize);
    if (listen(sock, backlog) < 0)
	msg_fatal("listen: %m");
    return (sock);
}

/* inet_accept - accept connection */

int     inet_accept(int fd)
{
    struct sockaddr_storage ss;
    SOCKADDR_SIZE ss_len = sizeof(ss);

    return (sane_accept(fd, (struct sockaddr *) &ss, &ss_len));
}
