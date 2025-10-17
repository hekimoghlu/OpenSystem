/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#ifndef COMPAT_GETADDRINFO_H
#define COMPAT_GETADDRINFO_H

#include <config.h>

/* Skip this entire file if a system getaddrinfo was detected. */
#ifndef HAVE_GETADDRINFO

/* OpenBSD likes to have sys/types.h included before sys/socket.h. */
#include <sys/types.h>
#include <sys/socket.h>

/* The struct returned by getaddrinfo, from RFC 3493. */
struct addrinfo {
    int ai_flags;               /* AI_PASSIVE, AI_CANONNAME, .. */
    int ai_family;              /* AF_xxx */
    int ai_socktype;            /* SOCK_xxx */
    int ai_protocol;            /* 0 or IPPROTO_xxx for IPv4 and IPv6 */
    socklen_t ai_addrlen;       /* Length of ai_addr */
    char *ai_canonname;         /* Canonical name for nodename */
    struct sockaddr *ai_addr;   /* Binary address */
    struct addrinfo *ai_next;   /* Next structure in linked list */
};

/* Constants for ai_flags from RFC 3493, combined with binary or. */
#define AI_PASSIVE      0x0001
#define AI_CANONNAME    0x0002
#define AI_NUMERICHOST  0x0004
#define AI_NUMERICSERV  0x0008
#define AI_V4MAPPED     0x0010
#define AI_ALL          0x0020
#define AI_ADDRCONFIG   0x0040

/* Error return codes from RFC 3493. */
#define EAI_AGAIN       1       /* Temporary name resolution failure */
#define EAI_BADFLAGS    2       /* Invalid value in ai_flags parameter */
#define EAI_FAIL        3       /* Permanent name resolution failure */
#define EAI_FAMILY      4       /* Address family not recognized */
#define EAI_MEMORY      5       /* Memory allocation failure */
#define EAI_NONAME      6       /* nodename or servname unknown */
#define EAI_SERVICE     7       /* Service not recognized for socket type */
#define EAI_SOCKTYPE    8       /* Socket type not recognized */
#define EAI_SYSTEM      9       /* System error occurred, see errno */
#define EAI_OVERFLOW    10      /* An argument buffer overflowed */

/* Function prototypes. */
sudo_dso_public int sudo_getaddrinfo(const char *nodename, const char *servname,
                const struct addrinfo *hints, struct addrinfo **res);
sudo_dso_public void sudo_freeaddrinfo(struct addrinfo *ai);
sudo_dso_public const char *sudo_gai_strerror(int ecode);

/* Map sudo_* to RFC 3493 names. */
#undef getaddrinfo
#define getaddrinfo(_a, _b, _c, _d) sudo_getaddrinfo((_a), (_b), (_c), (_d))
#undef freeaddrinfo
#define freeaddrinfo(_a) sudo_freeaddrinfo((_a))
#undef gai_strerror
#define gai_strerror(_a) sudo_gai_strerror((_a))

#endif /* !HAVE_GETADDRINFO */
#endif /* COMPAT_GETADDRINFO_H */
