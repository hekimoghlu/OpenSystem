/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
/* System library. */

#include <sys_defs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <sock_addr.h>

/* sock_addr_cmp_addr - compare addresses for equality */

int     sock_addr_cmp_addr(const struct sockaddr *sa,
			           const struct sockaddr *sb)
{
    if (sa->sa_family != sb->sa_family)
	return (sa->sa_family - sb->sa_family);

    /*
     * With IPv6 address structures, assume a non-hostile implementation that
     * stores the address as a contiguous sequence of bits. Any holes in the
     * sequence would invalidate the use of memcmp().
     */
    if (sa->sa_family == AF_INET) {
	return (SOCK_ADDR_IN_ADDR(sa).s_addr - SOCK_ADDR_IN_ADDR(sb).s_addr);
#ifdef HAS_IPV6
    } else if (sa->sa_family == AF_INET6) {
	return (memcmp((void *) &(SOCK_ADDR_IN6_ADDR(sa)),
		       (void *) &(SOCK_ADDR_IN6_ADDR(sb)),
		       sizeof(SOCK_ADDR_IN6_ADDR(sa))));
#endif
    } else {
	msg_panic("sock_addr_cmp_addr: unsupported address family %d",
		  sa->sa_family);
    }
}

/* sock_addr_cmp_port - compare ports for equality */

int     sock_addr_cmp_port(const struct sockaddr *sa,
			           const struct sockaddr *sb)
{
    if (sa->sa_family != sb->sa_family)
	return (sa->sa_family - sb->sa_family);

    if (sa->sa_family == AF_INET) {
	return (SOCK_ADDR_IN_PORT(sa) - SOCK_ADDR_IN_PORT(sb));
#ifdef HAS_IPV6
    } else if (sa->sa_family == AF_INET6) {
	return (SOCK_ADDR_IN6_PORT(sa) - SOCK_ADDR_IN6_PORT(sb));
#endif
    } else {
	msg_panic("sock_addr_cmp_port: unsupported address family %d",
		  sa->sa_family);
    }
}

/* sock_addr_in_loopback - determine if address is loopback */

int     sock_addr_in_loopback(const struct sockaddr *sa)
{
    unsigned long inaddr;

    if (sa->sa_family == AF_INET) {
	inaddr = ntohl(SOCK_ADDR_IN_ADDR(sa).s_addr);
	return (IN_CLASSA(inaddr)
		&& ((inaddr & IN_CLASSA_NET) >> IN_CLASSA_NSHIFT)
		== IN_LOOPBACKNET);
#ifdef HAS_IPV6
    } else if (sa->sa_family == AF_INET6) {
	return (IN6_IS_ADDR_LOOPBACK(&SOCK_ADDR_IN6_ADDR(sa)));
#endif
    } else {
	msg_panic("sock_addr_in_loopback: unsupported address family %d",
		  sa->sa_family);
    }
}
