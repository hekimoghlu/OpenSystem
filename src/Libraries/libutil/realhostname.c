/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#include <sys/cdefs.h>

#include <sys/param.h>
#include <sys/socket.h>

#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <stdio.h>
#include <string.h>

#include "libutil.h"

struct sockinet {
	u_char	si_len;
	u_char	si_family;
	u_short	si_port;
};

void trimdomain(char *_fullhost, size_t _hostsize);

int
realhostname(char *host, size_t hsize, const struct in_addr *ip)
{
	char trimmed[MAXHOSTNAMELEN];
	int result;
	struct hostent *hp;

	result = HOSTNAME_INVALIDADDR;
	hp = gethostbyaddr((const char *)ip, sizeof(*ip), AF_INET);

	if (hp != NULL) {
		strlcpy(trimmed, hp->h_name, sizeof(trimmed));
		trimdomain(trimmed, strlen(trimmed));
		if (strlen(trimmed) <= hsize) {
			char lookup[MAXHOSTNAMELEN];

			strncpy(lookup, hp->h_name, sizeof(lookup) - 1);
			lookup[sizeof(lookup) - 1] = '\0';
			hp = gethostbyname(lookup);
			if (hp == NULL)
				result = HOSTNAME_INVALIDNAME;
			else for (; ; hp->h_addr_list++) {
				if (*hp->h_addr_list == NULL) {
					result = HOSTNAME_INCORRECTNAME;
					break;
				}
				if (!memcmp(*hp->h_addr_list, ip, sizeof(*ip))) {
					strncpy(host, trimmed, hsize);
					return HOSTNAME_FOUND;
				}
			}
		}
	}

	strncpy(host, inet_ntoa(*ip), hsize);

	return result;
}

int
realhostname_sa(char *host, size_t hsize, struct sockaddr *addr, int addrlen)
{
	int result, error;
	char buf[NI_MAXHOST];

	result = HOSTNAME_INVALIDADDR;

#ifdef INET6
	/* IPv4 mapped IPv6 addr consideraton, specified in rfc2373. */
	if (addr->sa_family == AF_INET6 &&
	    addrlen == sizeof(struct sockaddr_in6) &&
	    IN6_IS_ADDR_V4MAPPED(&((struct sockaddr_in6 *)addr)->sin6_addr)) {
		struct sockaddr_in6 *sin6;

		sin6 = (struct sockaddr_in6 *)addr;

		memset(&lsin, 0, sizeof(lsin));
		lsin.sin_len = sizeof(struct sockaddr_in);
		lsin.sin_family = AF_INET;
		lsin.sin_port = sin6->sin6_port;
		memcpy(&lsin.sin_addr, &sin6->sin6_addr.s6_addr[12],
		       sizeof(struct in_addr));
		addr = (struct sockaddr *)&lsin;
		addrlen = lsin.sin_len;
	}
#endif

	error = getnameinfo(addr, addrlen, buf, sizeof(buf), NULL, 0,
			    NI_NAMEREQD);
	if (error == 0) {
		struct addrinfo hints, *res, *ores;
		struct sockaddr *sa;

		memset(&hints, 0, sizeof(struct addrinfo));
		hints.ai_family = addr->sa_family;
		hints.ai_flags = AI_CANONNAME | AI_PASSIVE;
		hints.ai_socktype = SOCK_STREAM;

		error = getaddrinfo(buf, NULL, &hints, &res);
		if (error) {
			result = HOSTNAME_INVALIDNAME;
			goto numeric;
		}
		for (ores = res; ; res = res->ai_next) {
			if (res == NULL) {
				freeaddrinfo(ores);
				result = HOSTNAME_INCORRECTNAME;
				goto numeric;
			}
			sa = res->ai_addr;
			if (sa == NULL) {
				freeaddrinfo(ores);
				result = HOSTNAME_INCORRECTNAME;
				goto numeric;
			}
			if (sa->sa_len == addrlen &&
			    sa->sa_family == addr->sa_family) {
				((struct sockinet *)sa)->si_port = ((struct sockinet *)addr)->si_port;
#ifdef INET6
				/*
				 * XXX: sin6_socpe_id may not been
				 * filled by DNS
				 */
				if (sa->sa_family == AF_INET6 &&
				    ((struct sockaddr_in6 *)sa)->sin6_scope_id == 0)
					((struct sockaddr_in6 *)sa)->sin6_scope_id = ((struct sockaddr_in6 *)addr)->sin6_scope_id;
#endif
				if (!memcmp(sa, addr, sa->sa_len)) {
					result = HOSTNAME_FOUND;
					if (ores->ai_canonname == NULL) {
						freeaddrinfo(ores);
						goto numeric;
					}
					strlcpy(buf, ores->ai_canonname,
						sizeof(buf));
					trimdomain(buf, hsize);
					if (strlen(buf) > hsize &&
					    addr->sa_family == AF_INET) {
						freeaddrinfo(ores);
						goto numeric;
					}
					strncpy(host, buf, hsize);
					break;
				}
			}
		}
		freeaddrinfo(ores);
	} else {
    numeric:
		if (getnameinfo(addr, addrlen, buf, sizeof(buf), NULL, 0,
				NI_NUMERICHOST) == 0)
			strncpy(host, buf, hsize);
	}

	return result;
}


