/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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
#include <netinet/in.h>  
#include <arpa/inet.h>
#ifdef NEED_RESOLV_H
# include <arpa/nameser.h>
# include <resolv.h>
#endif /* NEED_RESOLV_H */
#include <netdb.h>
#include <errno.h>

#include "sudoers.h"
#include "interfaces.h"

static struct interface_list interfaces = SLIST_HEAD_INITIALIZER(interfaces);

/*
 * Parse a space-delimited list of IP address/netmask pairs and
 * store in a list of interface structures.  Returns true on
 * success and false on parse error or memory allocation error.
 */
bool
set_interfaces(const char *ai)
{
    char *addrinfo, *addr, *mask, *last;
    struct interface *ifp;
    bool ret = false;
    debug_decl(set_interfaces, SUDOERS_DEBUG_NETIF);

    if ((addrinfo = strdup(ai)) == NULL)
	debug_return_bool(false);
    for (addr = strtok_r(addrinfo, " \t", &last); addr != NULL; addr = strtok_r(NULL, " \t", &last)) {
	/* Separate addr and mask. */
	if ((mask = strchr(addr, '/')) == NULL)
	    continue;
	*mask++ = '\0';

	/* Parse addr and store in list. */
	if ((ifp = calloc(1, sizeof(*ifp))) == NULL) {
	    sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	    goto done;
	}
	if (strchr(addr, ':')) {
	    /* IPv6 */
#ifdef HAVE_STRUCT_IN6_ADDR
	    ifp->family = AF_INET6;
	    if (inet_pton(AF_INET6, addr, &ifp->addr.ip6) != 1) {
		sudo_warnx(U_("unable to parse IP address \"%s\""), addr);
		free(ifp);
		goto done;
	    }
	    if (inet_pton(AF_INET6, mask, &ifp->netmask.ip6) != 1) {
		sudo_warnx(U_("unable to parse netmask \"%s\""), mask);
		free(ifp);
		goto done;
	    }
#else
	    free(ifp);
	    continue;
#endif
	} else {
	    /* IPv4 */
	    ifp->family = AF_INET;
	    if (inet_pton(AF_INET, addr, &ifp->addr.ip4) != 1) {
		sudo_warnx(U_("unable to parse IP address \"%s\""), addr);
		free(ifp);
		goto done;
	    }
	    if (inet_pton(AF_INET, mask, &ifp->netmask.ip4) != 1) {
		sudo_warnx(U_("unable to parse netmask \"%s\""), mask);
		free(ifp);
		goto done;
	    }
	}
	SLIST_INSERT_HEAD(&interfaces, ifp, entries);
    }
    ret = true;

done:
    free(addrinfo);
    debug_return_bool(ret);
}

struct interface_list *
get_interfaces(void)
{
    return &interfaces;
}

void
dump_interfaces(const char *ai)
{
    const char *cp, *ep;
    const char *ai_end = ai + strlen(ai);
    debug_decl(set_interfaces, SUDOERS_DEBUG_NETIF);

    sudo_printf(SUDO_CONV_INFO_MSG,
	_("Local IP address and netmask pairs:\n"));
    cp = sudo_strsplit(ai, ai_end, " \t", &ep);
    while (cp != NULL) {
	sudo_printf(SUDO_CONV_INFO_MSG, "\t%.*s\n", (int)(ep - cp), cp);
	cp = sudo_strsplit(NULL, ai_end, " \t", &ep);
    }

    debug_return;
}
