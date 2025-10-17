/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

/*
 * Try to obtain a verified name for the address in `sa, salen' (much
 * similar to getnameinfo).
 * Verified in this context means that forwards and backwards lookups
 * in DNS are consistent.  If that fails, return an error if the
 * NI_NAMEREQD flag is set or return the numeric address as a string.
 */

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
getnameinfo_verified(const struct sockaddr *sa, socklen_t salen,
		     char *host, socklen_t hostlen,
		     char *serv, socklen_t servlen,
		     int flags)
{
    int ret;
    struct addrinfo *ai, *a;
    char servbuf[NI_MAXSERV];
    struct addrinfo hints;
    void *saaddr;
    size_t sasize;

    if (host == NULL)
	return EAI_NONAME;

    if (serv == NULL) {
	serv = servbuf;
	servlen = sizeof(servbuf);
    }

    ret = getnameinfo (sa, salen, host, hostlen, serv, servlen,
		       flags | NI_NUMERICSERV);
    if (ret)
	goto fail;

    memset (&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    ret = getaddrinfo (host, serv, &hints, &ai);
    if (ret)
	goto fail;

    saaddr = socket_get_address(sa);
    sasize = socket_addr_size(sa);
    for (a = ai; a != NULL; a = a->ai_next) {
	if (sasize == socket_addr_size(a->ai_addr) &&
	    memcmp(saaddr, socket_get_address(a->ai_addr), sasize) == 0) {
	    freeaddrinfo (ai);
	    return 0;
	}
    }
    freeaddrinfo (ai);
 fail:
    if (flags & NI_NAMEREQD)
	return EAI_NONAME;
    ret = getnameinfo (sa, salen, host, hostlen, serv, servlen,
		       flags | NI_NUMERICSERV | NI_NUMERICHOST);
    return ret;
}
