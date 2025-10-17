/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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

#ifndef HEADER_CURL_SETUP_WIN32_H
#define HEADER_CURL_SETUP_WIN32_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 * SPDX-License-Identifier: curl
 *
 ***************************************************************************/

#undef USE_WINSOCK
/* ---------------------------------------------------------------- */
/*                     Watt-32 TCP/IP SPECIFIC                      */
/* ---------------------------------------------------------------- */
#ifdef USE_WATT32
#  include <tcp.h>
#  undef byte
#  undef word
#  define HAVE_SYS_IOCTL_H
#  define HAVE_SYS_SOCKET_H
#  define HAVE_NETINET_IN_H
#  define HAVE_NETDB_H
#  define HAVE_ARPA_INET_H
#  define SOCKET int
/* ---------------------------------------------------------------- */
/*               BSD-style lwIP TCP/IP stack SPECIFIC               */
/* ---------------------------------------------------------------- */
#elif defined(USE_LWIPSOCK)
  /* Define to use BSD-style lwIP TCP/IP stack. */
  /* #define USE_LWIPSOCK 1 */
#  undef HAVE_GETHOSTNAME
#  undef LWIP_POSIX_SOCKETS_IO_NAMES
#  undef RECV_TYPE_ARG1
#  undef RECV_TYPE_ARG3
#  undef SEND_TYPE_ARG1
#  undef SEND_TYPE_ARG3
#  define HAVE_GETHOSTBYNAME_R
#  define HAVE_GETHOSTBYNAME_R_6
#  define LWIP_POSIX_SOCKETS_IO_NAMES 0
#  define RECV_TYPE_ARG1 int
#  define RECV_TYPE_ARG3 size_t
#  define SEND_TYPE_ARG1 int
#  define SEND_TYPE_ARG3 size_t
#elif defined(_WIN32)
#  define USE_WINSOCK 2
#endif

/*
 * Include header files for windows builds before redefining anything.
 * Use this preprocessor block only to include or exclude windows.h,
 * winsock2.h or ws2tcpip.h. Any other windows thing belongs
 * to any other further and independent block.  Under Cygwin things work
 * just as under linux (e.g. <sys/socket.h>) and the winsock headers should
 * never be included when __CYGWIN__ is defined.
 */

#ifdef _WIN32
#  if defined(UNICODE) && !defined(_UNICODE)
#    error "UNICODE is defined but _UNICODE is not defined"
#  endif
#  if defined(_UNICODE) && !defined(UNICODE)
#    error "_UNICODE is defined but UNICODE is not defined"
#  endif
/*
 * Don't include unneeded stuff in Windows headers to avoid compiler
 * warnings and macro clashes.
 * Make sure to define this macro before including any Windows headers.
 */
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOGDI
#    define NOGDI
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  include <winerror.h>
#  include <tchar.h>
#  ifdef UNICODE
     typedef wchar_t *(*curl_wcsdup_callback)(const wchar_t *str);
#  endif
#endif

/*
 * Define _WIN32_WINNT_[OS] symbols because not all Windows build systems have
 * those symbols to compare against, and even those that do may be missing
 * newer symbols.
 */

#ifndef _WIN32_WINNT_NT4
#define _WIN32_WINNT_NT4            0x0400   /* Windows NT 4.0 */
#endif
#ifndef _WIN32_WINNT_WIN2K
#define _WIN32_WINNT_WIN2K          0x0500   /* Windows 2000 */
#endif
#ifndef _WIN32_WINNT_WINXP
#define _WIN32_WINNT_WINXP          0x0501   /* Windows XP */
#endif
#ifndef _WIN32_WINNT_WS03
#define _WIN32_WINNT_WS03           0x0502   /* Windows Server 2003 */
#endif
#ifndef _WIN32_WINNT_VISTA
#define _WIN32_WINNT_VISTA          0x0600   /* Windows Vista */
#endif
#ifndef _WIN32_WINNT_WS08
#define _WIN32_WINNT_WS08           0x0600   /* Windows Server 2008 */
#endif
#ifndef _WIN32_WINNT_WIN7
#define _WIN32_WINNT_WIN7           0x0601   /* Windows 7 */
#endif
#ifndef _WIN32_WINNT_WIN8
#define _WIN32_WINNT_WIN8           0x0602   /* Windows 8 */
#endif
#ifndef _WIN32_WINNT_WINBLUE
#define _WIN32_WINNT_WINBLUE        0x0603   /* Windows 8.1 */
#endif
#ifndef _WIN32_WINNT_WIN10
#define _WIN32_WINNT_WIN10          0x0A00   /* Windows 10 */
#endif

#endif /* HEADER_CURL_SETUP_WIN32_H */
