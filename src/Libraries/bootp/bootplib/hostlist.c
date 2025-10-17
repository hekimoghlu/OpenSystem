/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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
 * hostlist.c
 * - in-core host entry list manipulation routines
 * - these are used for storing the in-core version of the 
 *   file-based host list and the in-core ignore list
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <mach/boolean.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <netinet/in_systm.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <net/if.h>
#include <netinet/if_ether.h>
#include <arpa/inet.h>
#include <signal.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <syslog.h>
#include "hostlist.h"

void
hostinsert(struct hosts * * hosts, struct hosts * hp)
{
    hp->next = *hosts;
    hp->prev = NULL;
    if (*hosts)
	(*hosts)->prev = hp;
    *hosts = hp;
}

void
hostremove(struct hosts * * hosts, struct hosts * hp)
{
    if (hp->prev)
	hp->prev->next = hp->next;
    else
	*hosts = hp->next;
    if (hp->next)
	hp->next->prev = hp->prev;
}

void
hostfree(struct hosts * * hosts,
	 struct hosts *hp
	 )
{
    hostremove(hosts, hp);
    if (hp->hostname) {
	free(hp->hostname);
	hp->hostname = NULL;
    }
    if (hp->bootfile) {
	free(hp->bootfile);
	hp->bootfile = NULL;
    }
    free((char *)hp);
}

struct hosts * 
hostadd(struct hosts * * hosts, struct timeval * tv_p, int htype,
	char * haddr, int hlen, struct in_addr * iaddr_p, 
	char * hostname, char * bootfile)
{
    struct hosts * hp;

    hp = (struct hosts *)malloc(sizeof(*hp));
    if (!hp)
	return (NULL);
    bzero(hp, sizeof(*hp));
    if (tv_p)
	hp->tv = *tv_p;
    hp->htype = htype;
    hp->hlen = hlen;
    if (hlen > sizeof(hp->haddr)) {
	hlen = sizeof(hp->haddr);
    }
    bcopy(haddr, &hp->haddr, hlen);
    if (iaddr_p)
	hp->iaddr = *iaddr_p;
    if (hostname)
	hp->hostname = strdup(hostname);
    if (bootfile)
	hp->bootfile = strdup(bootfile);
    hostinsert(hosts, hp);
    return (hp);
}
