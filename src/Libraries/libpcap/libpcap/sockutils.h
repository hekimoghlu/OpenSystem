/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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
#ifndef __SOCKUTILS_H__
#define __SOCKUTILS_H__

#ifdef _MSC_VER
#pragma once
#endif

#include "pcap/socket.h"

#ifndef _WIN32
  /* UN*X */
  #include <unistd.h>	/* close() */

  /*!
   * \brief In Winsock, the close() call cannot be used on a socket;
   * closesocket() must be used.
   * We define closesocket() to be a wrapper around close() on UN*X,
   * so that it can be used on both platforms.
   */
  #define closesocket(a) close(a)
#endif

#include "sslutils.h"  // for SSL type, whatever that turns out to be

/*
 * MingW headers include this definition, but only for Windows XP and above.
 * MSDN states that this function is available for most versions on Windows.
 */
#if ((defined(__MINGW32__)) && (_WIN32_WINNT < 0x0501))
int WSAAPI getnameinfo(const struct sockaddr*,socklen_t,char*,DWORD,
	char*,DWORD,int);
#endif

/*
 * \defgroup SockUtils Cross-platform socket utilities (IPv4-IPv6)
 */

/*
 * \addtogroup SockUtils
 * \{
 */

/*
 * \defgroup ExportedStruct Exported Structures and Definitions
 */

/*
 * \addtogroup ExportedStruct
 * \{
 */

/****************************************************
 *                                                  *
 * Exported functions / definitions                 *
 *                                                  *
 ****************************************************/

/* 'checkonly' flag, into the rpsock_bufferize() */
#define SOCKBUF_CHECKONLY 1
/* no 'checkonly' flag, into the rpsock_bufferize() */
#define SOCKBUF_BUFFERIZE 0

/* no 'server' flag; it opens a client socket */
#define SOCKOPEN_CLIENT 0
/* 'server' flag; it opens a server socket */
#define SOCKOPEN_SERVER 1

/*
 * Flags for sock_recv().
 */
#define SOCK_RECEIVEALL_NO	0x00000000	/* Don't wait to receive all data */
#define SOCK_RECEIVEALL_YES	0x00000001	/* Wait to receive all data */

#define SOCK_EOF_ISNT_ERROR	0x00000000	/* Return 0 on EOF */
#define SOCK_EOF_IS_ERROR	0x00000002	/* Return an error on EOF */

#define SOCK_MSG_PEEK		0x00000004	/* Return data but leave it in the socket queue */

/*
 * \}
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * \defgroup ExportedFunc Exported Functions
 */

/*
 * \addtogroup ExportedFunc
 * \{
 */

int sock_init(char *errbuf, int errbuflen);
void sock_cleanup(void);
void sock_fmterror(const char *caller, int errcode, char *errbuf, int errbuflen);
void sock_geterror(const char *caller, char *errbuf, int errbufsize);
int sock_initaddress(const char *address, const char *port,
    struct addrinfo *hints, struct addrinfo **addrinfo,
    char *errbuf, int errbuflen);
int sock_recv(SOCKET sock, SSL *, void *buffer, size_t size, int receiveall,
    char *errbuf, int errbuflen);
int sock_recv_dgram(SOCKET sock, SSL *, void *buffer, size_t size,
    char *errbuf, int errbuflen);
SOCKET sock_open(struct addrinfo *addrinfo, int server, int nconn, char *errbuf, int errbuflen);
int sock_close(SOCKET sock, char *errbuf, int errbuflen);

int sock_send(SOCKET sock, SSL *, const char *buffer, size_t size,
    char *errbuf, int errbuflen);
int sock_bufferize(const char *buffer, int size, char *tempbuf, int *offset, int totsize, int checkonly, char *errbuf, int errbuflen);
int sock_discard(SOCKET sock, SSL *, int size, char *errbuf, int errbuflen);
int	sock_check_hostlist(char *hostlist, const char *sep, struct sockaddr_storage *from, char *errbuf, int errbuflen);
int sock_cmpaddr(struct sockaddr_storage *first, struct sockaddr_storage *second);

int sock_getmyinfo(SOCKET sock, char *address, int addrlen, char *port, int portlen, int flags, char *errbuf, int errbuflen);

int sock_getascii_addrport(const struct sockaddr_storage *sockaddr, char *address, int addrlen, char *port, int portlen, int flags, char *errbuf, int errbuflen);
int sock_present2network(const char *address, struct sockaddr_storage *sockaddr, int addr_family, char *errbuf, int errbuflen);

#ifdef __cplusplus
}
#endif

/*
 * \}
 */

/*
 * \}
 */

#endif
