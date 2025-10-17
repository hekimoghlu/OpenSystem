/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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
/* $Id: platform.h,v 1.7 2007/06/18 23:47:52 tbox Exp $ */

#ifndef LWRES_PLATFORM_H
#define LWRES_PLATFORM_H 1

/*****
 ***** Platform-dependent defines.
 *****/

/***
 *** Network.
 ***/

/*
 * Define if this system needs the <netinet/in6.h> header file for IPv6.
 */
/*@LWRES_PLATFORM_NEEDNETINETIN6H@ */

/*
 * Define if this system needs the <netinet6/in6.h> header file for IPv6.
 */
/*@LWRES_PLATFORM_NEEDNETINET6IN6H@ */

/*
 * If sockaddrs on this system have an sa_len field, LWRES_PLATFORM_HAVESALEN
 * will be defined.
 */
/*@LWRES_PLATFORM_HAVESALEN@ */

/*
 * If this system has the IPv6 structure definitions, LWRES_PLATFORM_HAVEIPV6
 * will be defined.
 */
/*@LWRES_PLATFORM_HAVEIPV6@ */

/*
 * If this system is missing in6addr_any, LWRES_PLATFORM_NEEDIN6ADDRANY will
 * be defined.
 */
#define LWRES_PLATFORM_NEEDIN6ADDRANY

/*
 * If this system has in_addr6, rather than in6_addr,
 * LWRES_PLATFORM_HAVEINADDR6 will be defined.
 */
/*@LWRES_PLATFORM_HAVEINADDR6@ */

/*
 * Defined if unistd.h does not cause fd_set to be delared.
 */
/*@LWRES_PLATFORM_NEEDSYSSELECTH@ */

/* VS2005 does not provide strlcpy() */
#define LWRES_PLATFORM_NEEDSTRLCPY

/*
 * Define some Macros
 */
#ifdef LIBLWRES_EXPORTS
#define LIBLWRES_EXTERNAL_DATA __declspec(dllexport)
#else
#define LIBLWRES_EXTERNAL_DATA __declspec(dllimport)
#endif

/*
 * Define the MAKE_NONBLOCKING Macro here since it can get used in
 * a number of places.
 */
#define MAKE_NONBLOCKING(sd, retval) \
do { \
	int _on = 1; \
	retval = ioctlsocket((SOCKET) sd, FIONBIO, &_on); \
} while (0)

/*
 * Need to define close here since lwres closes sockets and not files
 */
#undef  close
#define close closesocket

/*
 * Internal to liblwres.
 */
void InitSockets(void);

void DestroySockets(void);

#endif /* LWRES_PLATFORM_H */
