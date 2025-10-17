/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
 * Issues to be discussed:
 * - Thread safe-ness must be checked
 * - Return values.  There seems to be no standard for return value (RFC2133)
 *   but INRIA implementation returns EAI_xxx defined for getaddrinfo().
 */

#include "rsync.h"

#define SUCCESS 0
#define ANY 0
#define YES 1
#define NO  0

static struct afd {
	int a_af;
	int a_addrlen;
	int a_socklen;
	int a_off;
} afdl [] = {
#ifdef INET6
	{PF_INET6, sizeof(struct in6_addr), sizeof(struct sockaddr_in6),
		offsetof(struct sockaddr_in6, sin6_addr)},
#endif
	{PF_INET, sizeof(struct in_addr), sizeof(struct sockaddr_in),
		offsetof(struct sockaddr_in, sin_addr)},
	{0, 0, 0, 0},
};

struct sockinet {
	u_char	si_len;
	u_char	si_family;
	u_short	si_port;
};

#define ENI_NOSOCKET 	0
#define ENI_NOSERVNAME	1
#define ENI_NOHOSTNAME	2
#define ENI_MEMORY	3
#define ENI_SYSTEM	4
#define ENI_FAMILY	5
#define ENI_SALEN	6

int
getnameinfo(sa, salen, host, hostlen, serv, servlen, flags)
	const struct sockaddr *sa;
	size_t salen;
	char *host;
	size_t hostlen;
	char *serv;
	size_t servlen;
	int flags;
{
	extern int h_errno;
	struct afd *afd;
	struct servent *sp;
	struct hostent *hp;
	u_short port;
	int family, i;
	char *addr, *p;
	u_long v4a;
#ifdef INET6
	u_char pfx;
	int h_error;
#endif
	char numaddr[512];

	if (sa == NULL)
		return ENI_NOSOCKET;

#ifdef HAVE_SOCKADDR_LEN
	if (sa->sa_len != salen) return ENI_SALEN;
#endif
	
	family = sa->sa_family;
	for (i = 0; afdl[i].a_af; i++)
		if (afdl[i].a_af == family) {
			afd = &afdl[i];
			goto found;
		}
	return ENI_FAMILY;
	
 found:
	if (salen != (size_t)afd->a_socklen)
		return ENI_SALEN;
	
	port = ((struct sockinet *)sa)->si_port; /* network byte order */
	addr = (char *)sa + afd->a_off;

	if (serv == NULL || servlen == 0) {
		/* what we should do? */
	} else if (flags & NI_NUMERICSERV) {
		if ((size_t)snprintf(serv, servlen+1, "%d", ntohs(port)) > servlen)
			return ENI_MEMORY;
	} else {
		sp = getservbyport(port, (flags & NI_DGRAM) ? "udp" : "tcp");
		if (sp) {
			if (strlcpy(serv, sp->s_name, servlen + 1) > servlen)
				return ENI_MEMORY;
		} else
			return ENI_NOSERVNAME;
	}

	switch (sa->sa_family) {
	case AF_INET:
		v4a = ((struct sockaddr_in *)sa)->sin_addr.s_addr;
		if (IN_MULTICAST(v4a) || IN_EXPERIMENTAL(v4a))
			flags |= NI_NUMERICHOST;
		v4a >>= IN_CLASSA_NSHIFT;
		if (v4a == 0 || v4a == IN_LOOPBACKNET)
			flags |= NI_NUMERICHOST;			
		break;
#ifdef INET6
	case AF_INET6:
		pfx = ((struct sockaddr_in6 *)sa)->sin6_addr.s6_addr[0];
		if (pfx == 0 || pfx == 0xfe || pfx == 0xff)
			flags |= NI_NUMERICHOST;
		break;
#endif
	}
	if (host == NULL || hostlen == 0) {
		/* what should we do? */
	} else if (flags & NI_NUMERICHOST) {
		if (!inet_ntop(afd->a_af, addr, numaddr, sizeof(numaddr)))
			return ENI_SYSTEM;
		if (strlcpy(host, numaddr, hostlen + 1) > hostlen)
			return ENI_MEMORY;
	} else {
#ifdef INET6
		hp = getipnodebyaddr(addr, afd->a_addrlen, afd->a_af, &h_error);
#else
		hp = gethostbyaddr(addr, afd->a_addrlen, afd->a_af);
		/*h_error = h_errno;*/
#endif

		if (hp) {
			if (flags & NI_NOFQDN) {
				p = strchr(hp->h_name, '.');
				if (p) *p = '\0';
			}
			if (strlcpy(host, hp->h_name, hostlen + 1) > hostlen) {
#ifdef INET6
				freehostent(hp);
#endif
				return ENI_MEMORY;
			}
#ifdef INET6
			freehostent(hp);
#endif
		} else {
			if (flags & NI_NAMEREQD)
				return ENI_NOHOSTNAME;
			if (!inet_ntop(afd->a_af, addr, numaddr, sizeof(numaddr)))
				return ENI_NOHOSTNAME;
			if (strlcpy(host, numaddr, hostlen + 1) > hostlen)
				return ENI_MEMORY;
		}
	}
	return SUCCESS;
}
