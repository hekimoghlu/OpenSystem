/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
/* $Id: ipv6.h,v 1.16 2007/06/19 23:47:23 tbox Exp $ */

#ifndef LWRES_IPV6_H
#define LWRES_IPV6_H 1

/*****
 ***** Module Info
 *****/

/*! \file lwres/ipv6.h
 * IPv6 definitions for systems which do not support IPv6.
 */

/***
 *** Imports.
 ***/

#include <lwres/int.h>
#include <lwres/platform.h>

/***
 *** Types.
 ***/

/*% in6_addr structure */
struct in6_addr {
        union {
		lwres_uint8_t	_S6_u8[16];
		lwres_uint16_t	_S6_u16[8];
		lwres_uint32_t	_S6_u32[4];
        } _S6_un;
};
/*@{*/
/*% IP v6 types */
#define s6_addr		_S6_un._S6_u8
#define s6_addr8	_S6_un._S6_u8
#define s6_addr16	_S6_un._S6_u16
#define s6_addr32	_S6_un._S6_u32
/*@}*/

#define IN6ADDR_ANY_INIT 	{{{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 }}}
#define IN6ADDR_LOOPBACK_INIT 	{{{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 }}}

LIBLWRES_EXTERNAL_DATA extern const struct in6_addr in6addr_any;
LIBLWRES_EXTERNAL_DATA extern const struct in6_addr in6addr_loopback;

/*% used in getaddrinfo.c and getnameinfo.c */
struct sockaddr_in6 {
#ifdef LWRES_PLATFORM_HAVESALEN
	lwres_uint8_t		sin6_len;
	lwres_uint8_t		sin6_family;
#else
	lwres_uint16_t		sin6_family;
#endif
	lwres_uint16_t		sin6_port;
	lwres_uint32_t		sin6_flowinfo;
	struct in6_addr		sin6_addr;
	lwres_uint32_t		sin6_scope_id;
};

#ifdef LWRES_PLATFORM_HAVESALEN
#define SIN6_LEN 1
#endif

/*% in6_pktinfo structure */
struct in6_pktinfo {
	struct in6_addr ipi6_addr;    /*%< src/dst IPv6 address */
	unsigned int    ipi6_ifindex; /*%< send/recv interface index */
};

/*!
 * Unspecified IPv6 address
 */
#define IN6_IS_ADDR_UNSPECIFIED(a)      \
        (((a)->s6_addr32[0] == 0) &&    \
         ((a)->s6_addr32[1] == 0) &&    \
         ((a)->s6_addr32[2] == 0) &&    \
         ((a)->s6_addr32[3] == 0))

/*
 * Loopback
 */
#define IN6_IS_ADDR_LOOPBACK(a)         \
        (((a)->s6_addr32[0] == 0) &&    \
         ((a)->s6_addr32[1] == 0) &&    \
         ((a)->s6_addr32[2] == 0) &&    \
         ((a)->s6_addr32[3] == htonl(1)))

/*
 * IPv4 compatible
 */
#define IN6_IS_ADDR_V4COMPAT(a)         \
        (((a)->s6_addr32[0] == 0) &&    \
         ((a)->s6_addr32[1] == 0) &&    \
         ((a)->s6_addr32[2] == 0) &&    \
         ((a)->s6_addr32[3] != 0) &&    \
         ((a)->s6_addr32[3] != htonl(1)))

/*
 * Mapped
 */
#define IN6_IS_ADDR_V4MAPPED(a)               \
        (((a)->s6_addr32[0] == 0) &&          \
         ((a)->s6_addr32[1] == 0) &&          \
         ((a)->s6_addr32[2] == htonl(0x0000ffff)))

#endif /* LWRES_IPV6_H */
