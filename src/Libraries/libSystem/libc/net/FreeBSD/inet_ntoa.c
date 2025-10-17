/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#if defined(LIBC_SCCS) && !defined(lint)
static const char sccsid[] = "@(#)inet_ntoa.c	8.1 (Berkeley) 6/4/93";
static const char rcsid[] = "$Id: inet_ntoa.c,v 1.1.352.1 2005/04/27 05:00:54 sra Exp $";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/inet/inet_ntoa.c,v 1.6 2007/06/14 07:13:28 delphij Exp $");

#include "port_before.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <stdio.h>
#include <string.h>

#include "port_after.h"

/*%
 * Convert network-format internet address
 * to base 256 d.d.d.d representation.
 */
/*const*/ char *
inet_ntoa(struct in_addr in) {
	static char ret[18];

	strcpy(ret, "[inet_ntoa error]");
	(void) inet_ntop(AF_INET, &in, ret, sizeof ret);
	return (ret);
}

#if 0
char *
inet_ntoa_r(struct in_addr in, char *buf, socklen_t size)
{

	(void) inet_ntop(AF_INET, &in, buf, size);
	return (buf);
}
#endif

/*
 * Weak aliases for applications that use certain private entry points,
 * and fail to include <arpa/inet.h>.
 */
#undef inet_ntoa
__weak_reference(__inet_ntoa, inet_ntoa);
__weak_reference(__inet_ntoa_r, inet_ntoa_r);

/*! \file */
