/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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

#ifdef HAVE_WINSOCK

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
inet_pton(int af, const char *csrc, void *dst)
{
    char * src;

    if (csrc == NULL || (src = strdup(csrc)) == NULL) {
	_set_errno( ENOMEM );
	return 0;
    }

    switch (af) {
    case AF_INET:
	{
	    struct sockaddr_in  si4;
	    INT r;
	    INT s = sizeof(si4);

	    si4.sin_family = AF_INET;
	    r = WSAStringToAddress(src, AF_INET, NULL, (LPSOCKADDR) &si4, &s);
	    free(src);
	    src = NULL;

	    if (r == 0) {
		memcpy(dst, &si4.sin_addr, sizeof(si4.sin_addr));
		return 1;
	    }
	}
	break;

    case AF_INET6:
	{
	    struct sockaddr_in6 si6;
	    INT r;
	    INT s = sizeof(si6);

	    si6.sin6_family = AF_INET6;
	    r = WSAStringToAddress(src, AF_INET6, NULL, (LPSOCKADDR) &si6, &s);
	    free(src);
	    src = NULL;

	    if (r == 0) {
		memcpy(dst, &si6.sin6_addr, sizeof(si6.sin6_addr));
		return 1;
	    }
	}
	break;

    default:
	_set_errno( EAFNOSUPPORT );
	return -1;
    }

    /* the call failed */
    {
	int le = WSAGetLastError();

	if (le == WSAEINVAL)
	    return 0;

	_set_errno(le);
	return -1;
    }
}

#else  /* !HAVE_WINSOCK */

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
inet_pton(int af, const char *src, void *dst)
{
    if (af != AF_INET) {
	errno = EAFNOSUPPORT;
	return -1;
    }
    return inet_aton (src, dst);
}

#endif
