/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#ifndef ADDR_INFO_H
#define ADDR_INFO_H

/* special compatibility hack */
#undef EAI_ADDRFAMILY
#undef EAI_AGAIN
#undef EAI_BADFLAGS
#undef EAI_FAIL
#undef EAI_FAMILY
#undef EAI_MEMORY
#undef EAI_NODATA
#undef EAI_NONAME
#undef EAI_SERVICE
#undef EAI_SOCKTYPE
#undef EAI_SYSTEM
#undef EAI_BADHINTS
#undef EAI_PROTOCOL
#undef EAI_MAX

#undef AI_PASSIVE
#undef AI_CANONNAME
#undef AI_NUMERICHOST
#undef AI_NUMERICSERV
#undef AI_ALL
#undef AI_ADDRCONFIG
#undef AI_V4MAPPED
#undef AI_DEFAULT

#undef NI_NOFQDN
#undef NI_NUMERICHOST
#undef NI_NAMEREQD
#undef NI_NUMERICSERV
#undef NI_DGRAM

#ifndef __P
# ifdef HAVE_PROTOTYPES
#  define __P(args) args
# else
#  define __P(args) ()
# endif
#endif

/* special compatibility hack -- end*/


/*
 * Error return codes from getaddrinfo()
 */
#define	EAI_ADDRFAMILY	 1	/* address family for hostname not supported */
#define	EAI_AGAIN	 2	/* temporary failure in name resolution */
#define	EAI_BADFLAGS	 3	/* invalid value for ai_flags */
#define	EAI_FAIL	 4	/* non-recoverable failure in name resolution */
#define	EAI_FAMILY	 5	/* ai_family not supported */
#define	EAI_MEMORY	 6	/* memory allocation failure */
#define	EAI_NODATA	 7	/* no address associated with hostname */
#define	EAI_NONAME	 8	/* hostname nor servname provided, or not known */
#define	EAI_SERVICE	 9	/* servname not supported for ai_socktype */
#define	EAI_SOCKTYPE	10	/* ai_socktype not supported */
#define	EAI_SYSTEM	11	/* system error returned in errno */
#define EAI_BADHINTS	12
#define EAI_PROTOCOL	13
#define EAI_MAX		14

/*
 * Flag values for getaddrinfo()
 */
#define	AI_PASSIVE	0x00000001 /* get address to use bind() */
#define	AI_CANONNAME	0x00000002 /* fill ai_canonname */
#define	AI_NUMERICHOST	0x00000004 /* prevent name resolution */
#define	AI_NUMERICSERV	0x00000008 /* prevent service name resolution */
/* valid flags for addrinfo */
#ifndef __HAIKU__
#undef AI_MASK
#define	AI_MASK		(AI_PASSIVE | AI_CANONNAME | AI_NUMERICHOST | AI_NUMERICSERV)
#endif

#define	AI_ALL		0x00000100 /* IPv6 and IPv4-mapped (with AI_V4MAPPED) */
#define	AI_V4MAPPED_CFG	0x00000200 /* accept IPv4-mapped if kernel supports */
#define	AI_ADDRCONFIG	0x00000400 /* only if any address is assigned */
#define	AI_V4MAPPED	0x00000800 /* accept IPv4-mapped IPv6 address */
/* special recommended flags for getipnodebyname */
#define	AI_DEFAULT	(AI_V4MAPPED_CFG | AI_ADDRCONFIG)

/*
 * Constants for getnameinfo()
 */
#ifndef NI_MAXHOST
#define	NI_MAXHOST	1025
#define	NI_MAXSERV	32
#endif

/*
 * Flag values for getnameinfo()
 */
#define	NI_NOFQDN	0x00000001
#define	NI_NUMERICHOST	0x00000002
#define	NI_NAMEREQD	0x00000004
#define	NI_NUMERICSERV	0x00000008
#define	NI_DGRAM	0x00000010

#ifndef HAVE_TYPE_STRUCT_ADDRINFO
struct addrinfo {
	int	ai_flags;	/* AI_PASSIVE, AI_CANONNAME */
	int	ai_family;	/* PF_xxx */
	int	ai_socktype;	/* SOCK_xxx */
	int	ai_protocol;	/* 0 or IPPROTO_xxx for IPv4 and IPv6 */
	size_t	ai_addrlen;	/* length of ai_addr */
	char	*ai_canonname;	/* canonical name for hostname */
	struct sockaddr *ai_addr;	/* binary address */
	struct addrinfo *ai_next;	/* next structure in linked list */
};
#endif

#ifndef HAVE_GETADDRINFO
#undef getaddrinfo
#define getaddrinfo getaddrinfo__compat
#endif
#ifndef HAVE_GETNAMEINFO
#undef getnameinfo
#define getnameinfo getnameinfo__compat
#endif
#ifndef HAVE_FREEHOSTENT
#undef freehostent
#define freehostent freehostent__compat
#endif
#ifndef HAVE_FREEADDRINFO
#undef freeaddrinfo
#define freeaddrinfo freeaddrinfo__compat
#endif

extern int getaddrinfo __P((
	const char *hostname, const char *servname,
	const struct addrinfo *hints,
	struct addrinfo **res));

extern int getnameinfo __P((
	const struct sockaddr *sa,
	socklen_t salen,
	char *host,
	socklen_t hostlen,
	char *serv,
	socklen_t servlen,
	int flags));

extern void freehostent __P((struct hostent *));
extern void freeaddrinfo __P((struct addrinfo *));
extern
#ifdef GAI_STRERROR_CONST
const
#endif
char *gai_strerror __P((int));

/* In case there is no definition of offsetof() provided - though any proper
Standard C system should have one. */

#ifndef offsetof
#define offsetof(p_type,field) ((size_t)&(((p_type *)0)->field))
#endif

#endif
