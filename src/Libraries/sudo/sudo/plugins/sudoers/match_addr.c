/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#ifdef NEED_RESOLV_H
# include <arpa/nameser.h>
# include <resolv.h>
#endif /* NEED_RESOLV_H */

#include "sudoers.h"
#include "interfaces.h"

static bool
addr_matches_if(const char *n)
{
    union sudo_in_addr_un addr;
    struct interface *ifp;
#ifdef HAVE_STRUCT_IN6_ADDR
    unsigned int j;
#endif
    unsigned int family;
    debug_decl(addr_matches_if, SUDOERS_DEBUG_MATCH);

#ifdef HAVE_STRUCT_IN6_ADDR
    if (inet_pton(AF_INET6, n, &addr.ip6) == 1) {
	family = AF_INET6;
    } else
#endif /* HAVE_STRUCT_IN6_ADDR */
    if (inet_pton(AF_INET, n, &addr.ip4) == 1) {
	family = AF_INET;
    } else {
	debug_return_bool(false);
    }

    SLIST_FOREACH(ifp, get_interfaces(), entries) {
	if (ifp->family != family)
	    continue;
	switch (family) {
	    case AF_INET:
		if (ifp->addr.ip4.s_addr == addr.ip4.s_addr ||
		    (ifp->addr.ip4.s_addr & ifp->netmask.ip4.s_addr)
		    == addr.ip4.s_addr)
		    debug_return_bool(true);
		break;
#ifdef HAVE_STRUCT_IN6_ADDR
	    case AF_INET6:
		if (memcmp(ifp->addr.ip6.s6_addr, addr.ip6.s6_addr,
		    sizeof(addr.ip6.s6_addr)) == 0)
		    debug_return_bool(true);
		for (j = 0; j < sizeof(addr.ip6.s6_addr); j++) {
		    if ((ifp->addr.ip6.s6_addr[j] & ifp->netmask.ip6.s6_addr[j]) != addr.ip6.s6_addr[j])
			break;
		}
		if (j == sizeof(addr.ip6.s6_addr))
		    debug_return_bool(true);
		break;
#endif /* HAVE_STRUCT_IN6_ADDR */
	}
    }

    debug_return_bool(false);
}

static bool
addr_matches_if_netmask(const char *n, const char *m)
{
    unsigned int i;
    union sudo_in_addr_un addr, mask;
    struct interface *ifp;
#ifdef HAVE_STRUCT_IN6_ADDR
    unsigned int j;
#endif
    unsigned int family;
    const char *errstr;
    debug_decl(addr_matches_if, SUDOERS_DEBUG_MATCH);

#ifdef HAVE_STRUCT_IN6_ADDR
    if (inet_pton(AF_INET6, n, &addr.ip6) == 1)
	family = AF_INET6;
    else
#endif /* HAVE_STRUCT_IN6_ADDR */
    if (inet_pton(AF_INET, n, &addr.ip4) == 1) {
	family = AF_INET;
    } else {
	debug_return_bool(false);
    }

    if (family == AF_INET) {
	if (strchr(m, '.')) {
	    if (inet_pton(AF_INET, m, &mask.ip4) != 1) {
		sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO,
		    "IPv4 netmask %s: %s", m, "invalid value");
		debug_return_bool(false);
	    }
	} else {
	    i = sudo_strtonum(m, 1, 32, &errstr);
	    if (errstr != NULL) {
		sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO,
		    "IPv4 netmask %s: %s", m, errstr);
		debug_return_bool(false);
	    }
	    mask.ip4.s_addr = htonl(0xffffffffU << (32 - i));
	}
	addr.ip4.s_addr &= mask.ip4.s_addr;
    }
#ifdef HAVE_STRUCT_IN6_ADDR
    else {
	if (inet_pton(AF_INET6, m, &mask.ip6) != 1) {
	    j = sudo_strtonum(m, 1, 128, &errstr);
	    if (errstr != NULL) {
		sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO,
		    "IPv6 netmask %s: %s", m, errstr);
		debug_return_bool(false);
	    }
	    for (i = 0; i < sizeof(addr.ip6.s6_addr); i++) {
		if (j < i * 8)
		    mask.ip6.s6_addr[i] = 0;
		else if (i * 8 + 8 <= j)
		    mask.ip6.s6_addr[i] = 0xff;
		else
		    mask.ip6.s6_addr[i] = 0xff00 >> (j - i * 8);
		addr.ip6.s6_addr[i] &= mask.ip6.s6_addr[i];
	    }
	}
    }
#endif /* HAVE_STRUCT_IN6_ADDR */

    SLIST_FOREACH(ifp, get_interfaces(), entries) {
	if (ifp->family != family)
	    continue;
	switch (family) {
	    case AF_INET:
		if ((ifp->addr.ip4.s_addr & mask.ip4.s_addr) == addr.ip4.s_addr)
		    debug_return_bool(true);
		break;
#ifdef HAVE_STRUCT_IN6_ADDR
	    case AF_INET6:
		for (j = 0; j < sizeof(addr.ip6.s6_addr); j++) {
		    if ((ifp->addr.ip6.s6_addr[j] & mask.ip6.s6_addr[j]) != addr.ip6.s6_addr[j])
			break;
		}
		if (j == sizeof(addr.ip6.s6_addr))
		    debug_return_bool(true);
		break;
#endif /* HAVE_STRUCT_IN6_ADDR */
	}
    }

    debug_return_bool(false);
}

/*
 * Returns true if "n" is one of our ip addresses or if
 * "n" is a network that we are on, else returns false.
 */
bool
addr_matches(char *n)
{
    char *m;
    bool rc;
    debug_decl(addr_matches, SUDOERS_DEBUG_MATCH);

    /* If there's an explicit netmask, use it. */
    if ((m = strchr(n, '/'))) {
	*m++ = '\0';
	rc = addr_matches_if_netmask(n, m);
	*(m - 1) = '/';
    } else
	rc = addr_matches_if(n);

    sudo_debug_printf(SUDO_DEBUG_DEBUG|SUDO_DEBUG_LINENO,
	"IP address %s matches local host: %s", n, rc ? "true" : "false");
    debug_return_bool(rc);
}
