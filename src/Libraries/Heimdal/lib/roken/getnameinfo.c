/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include <config.h>

#include "roken.h"

static int
doit (int af,
      const void *addr,
      size_t addrlen,
      int port,
      char *host, size_t hostlen,
      char *serv, size_t servlen,
      int flags)
{
    if (host != NULL) {
	if (flags & NI_NUMERICHOST) {
	    if (inet_ntop (af, addr, host, hostlen) == NULL)
		return EAI_SYSTEM;
	} else {
	    struct hostent *he = gethostbyaddr (addr,
						addrlen,
						af);
	    if (he != NULL) {
		strlcpy (host, hostent_find_fqdn(he), hostlen);
		if (flags & NI_NOFQDN) {
		    char *dot = strchr (host, '.');
		    if (dot != NULL)
			*dot = '\0';
		}
	    } else if (flags & NI_NAMEREQD) {
		return EAI_NONAME;
	    } else if (inet_ntop (af, addr, host, hostlen) == NULL)
		return EAI_SYSTEM;
	}
    }

    if (serv != NULL) {
	if (flags & NI_NUMERICSERV) {
	    snprintf (serv, servlen, "%u", ntohs(port));
	} else {
	    const char *proto = "tcp";
	    struct servent *se;

	    if (flags & NI_DGRAM)
		proto = "udp";

	    se = getservbyport (port, proto);
	    if (se == NULL) {
		snprintf (serv, servlen, "%u", ntohs(port));
	    } else {
		strlcpy (serv, se->s_name, servlen);
	    }
	}
    }
    return 0;
}

/*
 *
 */

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
getnameinfo(const struct sockaddr *sa, socklen_t salen,
	    char *host, size_t hostlen,
	    char *serv, size_t servlen,
	    int flags)
{
    switch (sa->sa_family) {
#ifdef HAVE_IPV6
    case AF_INET6 : {
	const struct sockaddr_in6 *sin6 = (const struct sockaddr_in6 *)sa;

	return doit (AF_INET6, &sin6->sin6_addr, sizeof(sin6->sin6_addr),
		     sin6->sin6_port,
		     host, hostlen,
		     serv, servlen,
		     flags);
    }
#endif
    case AF_INET : {
	const struct sockaddr_in *sin4 = (const struct sockaddr_in *)sa;

	return doit (AF_INET, &sin4->sin_addr, sizeof(sin4->sin_addr),
		     sin4->sin_port,
		     host, hostlen,
		     serv, servlen,
		     flags);
    }
    default :
	return EAI_FAMILY;
    }
}
