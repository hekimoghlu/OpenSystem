/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
/*
 * We need to include the zsh headers later to avoid clashes with
 * the definitions on some systems, however we need the configuration
 * file to decide whether we can include netinet/in_systm.h, which
 * doesn't exist on cygwin.
 */
#include "../../config.h"

#include <sys/types.h>
#include <sys/socket.h>

#ifdef HAVE_BIND_NETDB_H
/*
 * On systems where we're using -lbind, this has more definitions
 * than the standard header.
 */
#include <bind/netdb.h>
#else
#include <netdb.h>
#endif

/*
 * For some reason, configure doesn't always detect netinet/in_systm.h.
 * On some systems, including linux, this seems to be because gcc is
 * throwing up a warning message about the redefinition of
 * __USE_LARGEFILE.  This means the problem is somewhere in the
 * header files where we can't get at it.  For now, revert to
 * not including this file only on systems where we know it's missing.
 * Currently this is just some older versions of cygwin.
 */
#if defined(HAVE_NETINET_IN_SYSTM_H) || !defined(__CYGWIN__)
# include <netinet/in_systm.h>
#endif
#include <netinet/in.h>
#include <netinet/ip.h>
#include <arpa/inet.h>

/* Is IPv6 supported by the library? */

#if defined(AF_INET6) && defined(IN6ADDR_LOOPBACK_INIT) \
	&& defined(HAVE_INET_NTOP) && defined(HAVE_INET_PTON)
# define SUPPORT_IPV6 1
#endif

union tcp_sockaddr {
    struct sockaddr a;
    struct sockaddr_in in;
#ifdef SUPPORT_IPV6
    struct sockaddr_in6 in6;
#endif
};

typedef struct tcp_session *Tcp_session;

#define ZTCP_LISTEN  1
#define ZTCP_INBOUND 2
#define ZTCP_ZFTP    16

struct tcp_session {
    int fd;				/* file descriptor */
    union tcp_sockaddr sock;  	/* local address   */
    union tcp_sockaddr peer;  	/* remote address  */
    int flags;
};

#include "tcp.pro"

#ifndef INET_ADDRSTRLEN
# define INET_ADDRSTRLEN 16
#endif

#ifndef INET6_ADDRSTRLEN
# define INET6_ADDRSTRLEN 46
#endif
