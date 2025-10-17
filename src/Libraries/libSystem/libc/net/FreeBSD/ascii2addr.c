/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/net/ascii2addr.c,v 1.4 2002/03/22 21:52:28 obrien Exp $");

#include <sys/types.h>
#include <sys/socket.h>

#include <errno.h>
#include <string.h>

#include <net/if_dl.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int
ascii2addr(af, ascii, result)
	int af;
	const char *ascii;
	void *result;
{
	struct in_addr *ina;
	char strbuf[4*sizeof("123")]; /* long enough for V4 only */

	switch(af) {
	case AF_INET:
		ina = result;
		strbuf[0] = '\0';
		strncat(strbuf, ascii, (sizeof strbuf)-1);
		if (inet_aton(strbuf, ina))
			return sizeof(struct in_addr);
		errno = EINVAL;
		break;

	case AF_LINK:
		link_addr(ascii, result);
		/* oops... no way to detect failure */
		return sizeof(struct sockaddr_dl);

	default:
		errno = EPROTONOSUPPORT;
		break;
	}

	return -1;
}
