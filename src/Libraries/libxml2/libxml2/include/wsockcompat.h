/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#if !defined __XML_WSOCKCOMPAT_H__
#define __XML_WSOCKCOMPAT_H__

#ifdef _WIN32_WCE
#include <winsock.h>
#else
#include <errno.h>
#include <winsock2.h>

/* Fix for old MinGW. */
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif

/* the following is a workaround a problem for 'inline' keyword in said
   header when compiled with Borland C++ 6 */
#if defined(__BORLANDC__) && !defined(__cplusplus)
#define inline __inline
#define _inline __inline
#endif

#include <ws2tcpip.h>

/* Check if ws2tcpip.h is a recent version which provides getaddrinfo() */
#if defined(GetAddrInfo)
#include <wspiapi.h>
#define HAVE_GETADDRINFO
#endif
#endif

#undef XML_SOCKLEN_T
#define XML_SOCKLEN_T int

#ifndef ECONNRESET
#define ECONNRESET WSAECONNRESET
#endif
#ifndef EINPROGRESS
#define EINPROGRESS WSAEINPROGRESS
#endif
#ifndef EINTR
#define EINTR WSAEINTR
#endif
#ifndef ESHUTDOWN
#define ESHUTDOWN WSAESHUTDOWN
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK WSAEWOULDBLOCK
#endif

#endif /* __XML_WSOCKCOMPAT_H__ */
