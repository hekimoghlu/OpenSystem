/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
/* $Id$ */

#ifndef ISC_IPV6_H
#define ISC_IPV6_H 1

/*****
 ***** Module Info
 *****/

/*
 * IPv6 definitions for systems which do not support IPv6.
 *
 * MP:
 *	No impact.
 *
 * Reliability:
 *	No anticipated impact.
 *
 * Resources:
 *	N/A.
 *
 * Security:
 *	No anticipated impact.
 *
 * Standards:
 *	RFC2553.
 */

#if _MSC_VER < 1300
#define in6_addr in_addr6
#endif

#ifndef IN6ADDR_ANY_INIT
#define IN6ADDR_ANY_INIT 	{{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 }}
#endif
#ifndef IN6ADDR_LOOPBACK_INIT
#define IN6ADDR_LOOPBACK_INIT 	{{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 }}
#endif
#ifndef IN6ADDR_V4MAPPED_INIT
#define IN6ADDR_V4MAPPED_INIT 	{{ 0,0,0,0,0,0,0,0,0,0,0xff,0xff,0,0,0,0 }}
#endif

LIBISC_EXTERNAL_DATA extern const struct in6_addr isc_in6addr_any;
LIBISC_EXTERNAL_DATA extern const struct in6_addr isc_in6addr_loopback;

/*
 * Unspecified
 */
#ifndef IN6_IS_ADDR_UNSPECIFIED
#define IN6_IS_ADDR_UNSPECIFIED(a) (\
*((u_long *)((a)->s6_addr)    ) == 0 && \
*((u_long *)((a)->s6_addr) + 1) == 0 && \
*((u_long *)((a)->s6_addr) + 2) == 0 && \
*((u_long *)((a)->s6_addr) + 3) == 0 \
)
#endif

/*
 * Loopback
 */
#ifndef IN6_IS_ADDR_LOOPBACK
#define IN6_IS_ADDR_LOOPBACK(a) (\
*((u_long *)((a)->s6_addr)    ) == 0 && \
*((u_long *)((a)->s6_addr) + 1) == 0 && \
*((u_long *)((a)->s6_addr) + 2) == 0 && \
*((u_long *)((a)->s6_addr) + 3) == htonl(1) \
)
#endif

/*
 * IPv4 compatible
 */
#define IN6_IS_ADDR_V4COMPAT(a)  (\
*((u_long *)((a)->s6_addr)    ) == 0 && \
*((u_long *)((a)->s6_addr) + 1) == 0 && \
*((u_long *)((a)->s6_addr) + 2) == 0 && \
*((u_long *)((a)->s6_addr) + 3) != 0 && \
*((u_long *)((a)->s6_addr) + 3) != htonl(1) \
)

/*
 * Mapped
 */
#define IN6_IS_ADDR_V4MAPPED(a) (\
*((u_long *)((a)->s6_addr)    ) == 0 && \
*((u_long *)((a)->s6_addr) + 1) == 0 && \
*((u_long *)((a)->s6_addr) + 2) == htonl(0x0000ffff))

/*
 * Multicast
 */
#define IN6_IS_ADDR_MULTICAST(a)	\
	((a)->s6_addr[0] == 0xffU)

/*
 * Unicast link / site local.
 */
#ifndef IN6_IS_ADDR_LINKLOCAL
#define IN6_IS_ADDR_LINKLOCAL(a)	(\
       ((a)->s6_addr[0] == 0xfe) && \
       (((a)->s6_addr[1] & 0xc0) == 0x80))
#endif

#ifndef IN6_IS_ADDR_SITELOCAL
#define IN6_IS_ADDR_SITELOCAL(a)	(\
       ((a)->s6_addr[0] == 0xfe) && \
       (((a)->s6_addr[1] & 0xc0) == 0xc0))
#endif

#endif /* ISC_IPV6_H */
