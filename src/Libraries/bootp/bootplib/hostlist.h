/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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
 * hostlist.h
 * - definitions for host list structures and functions
 */

#ifndef _S_HOSTLIST_H
#define _S_HOSTLIST_H

#include <netinet/in.h>

struct hosts {
	struct hosts	*next;
	struct hosts	*prev;
	struct in_addr	iaddr;		/* internet address */
	u_char		htype;		/* hardware type */
	u_char		hlen;		/* hardware length */
	union {				/* hardware address */
	    struct ether_addr 	en;
	    u_char		generic[256];
	} haddr;
	char *		hostname;	/* host name (and suffix) */
	char *		bootfile;	/* default boot file name */
	struct timeval	tv;		/* time-in */

        u_long		lease;		/* lease (dhcp only) */
};

struct hosts * 	hostadd(struct hosts * * hosts, struct timeval * tv_p, 
			int htype, char * haddr, int hlen, 
			struct in_addr * iaddr_p, char * host_name,
			char * bootfile);
void		hostfree(struct hosts * * hosts, struct hosts * hp);
void		hostinsert(struct hosts * * hosts, struct hosts * hp);
void		hostremove(struct hosts * * hosts, struct hosts * hp);

typedef boolean_t subnet_match_func_t(void * arg, struct in_addr iaddr);

static __inline__ struct hosts *
hostbyip(struct hosts * hosts, struct in_addr iaddr)
{
    struct hosts * hp;
    for (hp = hosts; hp; hp = hp->next) {
	if (iaddr.s_addr == hp->iaddr.s_addr)
	    return (hp);
    }
    return (NULL);
}

static __inline__ struct hosts *
hostbyaddr(struct hosts * hosts, u_char hwtype, void * hwaddr, int hwlen,
	   subnet_match_func_t * func, void * arg)
{
    struct hosts * hp;

    for (hp = hosts; hp; hp = hp->next) {
	if (hwtype == hp->htype 
	    && hwlen == hp->hlen
	    && bcmp(hwaddr, &hp->haddr, hwlen) == 0) {
	    if (func == NULL
		|| (*func)(arg, hp->iaddr)) {
		return (hp);
	    }
	}
    }
    return (NULL);
}


#endif /* _S_HOSTLIST_H */
