/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
static const char sccsid[] = "@(#)inet_makeaddr.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/inet/inet_makeaddr.c,v 1.4 2007/06/03 17:20:26 ume Exp $");

#include "port_before.h"

#include <sys/param.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "port_after.h"

/*%
 * Formulate an Internet address from network + host.  Used in
 * building addresses stored in the ifnet structure.
 */
struct in_addr
inet_makeaddr(in_addr_t net, in_addr_t host)
{
	struct in_addr a;

	if (net < 128U)
		a.s_addr = (net << IN_CLASSA_NSHIFT) | (host & IN_CLASSA_HOST);
	else if (net < 65536U)
		a.s_addr = (net << IN_CLASSB_NSHIFT) | (host & IN_CLASSB_HOST);
	else if (net < 16777216L)
		a.s_addr = (net << IN_CLASSC_NSHIFT) | (host & IN_CLASSC_HOST);
	else
		a.s_addr = net | host;
	a.s_addr = htonl(a.s_addr);
	return (a);
}

/*
 * Weak aliases for applications that use certain private entry points,
 * and fail to include <arpa/inet.h>.
 */
#undef inet_makeaddr
__weak_reference(__inet_makeaddr, inet_makeaddr);

/*! \file */
